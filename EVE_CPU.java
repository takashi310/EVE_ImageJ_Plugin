import ij.*;
import ij.io.*;
import ij.process.*;
import ij.gui.*;
import ij.measure.*;
import java.awt.*;
import ij.plugin.*;
import ij.plugin.filter.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.io.FileUtils;

import java.io.IOException;
import java.io.File;
import java.nio.ByteOrder;
import java.nio.Buffer;
import java.nio.FloatBuffer;

public class EVE_CPU implements PlugInFilter {
	ImagePlus imp_;
	int rad_  = (int)Prefs.get("eve_rad.int", 3);
	int th_  = (int)Prefs.get("eve_th.int", 20);
	int thread_num_ = (int)Prefs.get("eve_thread_num.int", 8);
	int ch_;
	int fr_;
	
	class LMaxima {
		public int x, y, z;
		public float score;
		LMaxima(int x_, int y_, int z_, float sc_){
			 x = x_; y = y_; z = z_; score = sc_;
		}
	}
		
	public int setup(String arg, ImagePlus imp) {
		this.imp_ = imp;
		imp.killRoi();
		this.ch_ = imp.getChannel();
		this.fr_ = imp.getFrame();
		return DOES_8G + DOES_16 + DOES_32;
	}

	private boolean showDialog() {
		GenericDialog gd = new GenericDialog("EVE GPU");
		gd.addNumericField("radius",  rad_, 0);
		gd.addNumericField("threshold",  th_, 0);
		gd.addNumericField("thread (CPU)",  thread_num_, 0);
		gd.showDialog();
	
		if (gd.wasCanceled()) return false;
		rad_ = (int)gd.getNextNumber();
		th_ = (int)gd.getNextNumber();
		thread_num_ = (int)gd.getNextNumber();
								
		Prefs.set("eve_rad.int", rad_);
		Prefs.set("eve_th.int", th_);
		Prefs.set("eve_thread_num.int", thread_num_);
		
		return true;
	}
		
	public void run(ImageProcessor ip) {
		if (!showDialog()) return;

		final float th = th_;

		int[] dims = imp_.getDimensions();
		final int imageW = dims[0];
		final int imageH = dims[1];
		final int nCh    = dims[2];
		final int imageD = dims[3];
		final int nFrame = dims[4];
		final int bdepth = imp_.getBitDepth();

		FileInfo finfo = imp_.getFileInfo();
		if(finfo == null) return;
		double xspc = finfo.pixelWidth;
		double yspc = finfo.pixelHeight;
		double zspc = finfo.pixelDepth;

		float xzratio = (float)(xspc/zspc);
		if (xzratio > 1.0f) return;
		if (rad_*xzratio < 1.0f) xzratio = 1.0f/rad_;
		final int r = rad_;
		final int zr = (int)(rad_*xzratio) > 0 ? (int)(rad_*xzratio) : 1;
		final float zfac = r*xzratio >= 1.0f ? (float)(zspc/xspc) : 1.0f/r;

		final int binnum = 255;
		int [] hist = new int[binnum];

		int[][][] mask = new int[zr+1][r+1][r+1];
		double rr = rad_*rad_;
		for (int dz = 0; dz <= zr; dz++){
			for (int dy = 0; dy <= rad_; dy++){
				for (int dx = 0; dx <= rad_; dx++){
					double dd = dx*dx + dy*dy + dz/xzratio*dz/xzratio;
					mask[dz][dy][dx] = dd <= rr ? 1 : 0;
				}
			}
		}

		int[][][] mask2 = new int[2*zr+1][2*r+1][2*r+1];
		for (int dz = 0; dz <= 2*zr; dz++){
			for (int dy = 0; dy <= 2*rad_; dy++){
				for (int dx = 0; dx <= 2*rad_; dx++){
					double dd = dx*dx + dy*dy + dz/xzratio*dz/xzratio;
					mask2[dz][dy][dx] = dd <= 4*rr ? 1 : 0;
				}
			}
		}
		ImagePlus tmpimp1 = IJ.createHyperStack("temp1", imageW, imageH, 1, imageD, 1, 32);
		ImagePlus tmpimp2 = IJ.createHyperStack("temp2", imageW, imageH, 1, imageD, 1, 32);
		
		ImagePlus newimp = IJ.createHyperStack("result", imageW, imageH, 2, imageD, 1, 32);
		
		ImageStack istack = imp_.getStack();
		ImageStack ostack = newimp.getStack();

		ImageStack tstack1 = tmpimp1.getStack();
		ImageStack tstack2 = tmpimp2.getStack();
	    
	    final ImageProcessor[] iplist = new ImageProcessor[imageD];
		final ImageProcessor[] oplist = new ImageProcessor[imageD];
		final ImageProcessor[] tplist1 = new ImageProcessor[imageD];
		final ImageProcessor[] tplist2 = new ImageProcessor[imageD];
		for(int s = 0; s < imageD; s++){
			iplist[s] =  istack.getProcessor(imp_.getStackIndex(ch_, s+1, fr_));
			oplist[s] =  ostack.getProcessor(newimp.getStackIndex(1, s+1, 1));
			tplist1[s] =  tstack1.getProcessor(tmpimp1.getStackIndex(1, s+1, 1));
			tplist2[s] =  tstack2.getProcessor(tmpimp2.getStackIndex(1, s+1, 1));
		}

		IJ.showProgress(0, 5);

		//calculate scores
		{
		final AtomicInteger ai1 = new AtomicInteger(zr);
		final Thread[] threads = newThreadArray();
		for (int ithread = 0; ithread < threads.length; ithread++) {
			// Concurrently run in as many threads as CPUs
			threads[ithread] = new Thread() {
		
				{ setPriority(Thread.NORM_PRIORITY); }
		
				public void run() {
					int rr = r*r;
					for (int z = ai1.getAndIncrement(); z < imageD-zr; z = ai1.getAndIncrement()) {
						for(int y = r; y < imageH-r; y++) {
							for(int x = r; x < imageW-r; x++) {
								int id = y * imageW + x;
								float sum = 0.0f;
								float center = iplist[z].getf(id);
								if (center >= th) {
									for (int dz = -zr; dz <= zr; dz++){
										for (int dy = -r; dy <= r; dy++){
											for (int dx = -r; dx <= r; dx++){
												if (dx*dx+dy*dy+dz*zfac*dz*zfac <= rr)
													sum += iplist[z+dz].getf(id+dy*imageW+dx);
											}
										}
									}
								}
								tplist1[z].setf(id, sum/((2*r+1)*(2*r+1)*(2*zr+1)));
							}
						}					
					}//	for (int i = ai.getAndIncrement(); i < names.length;
				}
			};//threads[ithread] = new Thread() {
		}//	for (int ithread = 0; ithread < threads.length; ithread++)
		startAndJoin(threads);
		}

/*		{
		final AtomicInteger ai1 = new AtomicInteger(2*zr);
		final Thread[] threads = newThreadArray();
		for (int ithread = 0; ithread < threads.length; ithread++) {
			// Concurrently run in as many threads as CPUs
			threads[ithread] = new Thread() {
		
				{ setPriority(Thread.NORM_PRIORITY); }
		
				public void run() {
					int rr = r*r;
					for (int z = ai1.getAndIncrement(); z < imageD-2*zr; z = ai1.getAndIncrement()) {
						for(int y = 2*r; y < imageH-2*r; y++) {
							for(int x = 2*r; x < imageW-2*r; x++) {
								int id = y * imageW + x;
								float sum = 0.0f;
								float center = iplist[z].getf(id);
								if (center >= th) {
									for (int dz = -zr; dz <= zr; dz++){
										for (int dy = -r; dy <= r; dy++){
											for (int dx = -r; dx <= r; dx++){
												if (dx*dx+dy*dy+dz*zfac*dz*zfac <= rr)
													sum += iplist[z+dz].getf(id+dy*imageW+dx);
											}
										}
									}
								}
								float mini = 65535.0f;
								if (center >= th) {
									for (int dz = -2*zr; dz <= 2*zr; dz++){
										for (int dy = -2*r; dy <= 2*r; dy++){
											for (int dx = -2*r; dx <= 2*r; dx++){
												float val = iplist[z+dz].getf(id+dy*imageW+dx);
												if (dx*dx+dy*dy+dz*zfac*dz*zfac <= 4*rr && val < mini)
													mini = val;
											}
										}
									}
								}
								tplist1[z].setf(id, sum/((2*r+1)*(2*r+1)*(2*zr+1)) - mini);
							}
						}					
					}//	for (int i = ai.getAndIncrement(); i < names.length;
				}
			};//threads[ithread] = new Thread() {
		}//	for (int ithread = 0; ithread < threads.length; ithread++)
		startAndJoin(threads);
		}
*/		
		IJ.showProgress(1, 5);

		//get local maxima
		{
		final AtomicInteger ai1 = new AtomicInteger(1);
		final Thread[] threads = newThreadArray();
		for (int ithread = 0; ithread < threads.length; ithread++) {
			// Concurrently run in as many threads as CPUs
			threads[ithread] = new Thread() {
		
				{ setPriority(Thread.NORM_PRIORITY); }
		
				public void run() {
					for (int z = ai1.getAndIncrement(); z < imageD-1; z = ai1.getAndIncrement()) {
						for(int y = 1; y < imageH-1; y++) {
							for(int x = 1; x < imageW-1; x++) {
								int id = y * imageW + x;
								boolean ismaxima = true;
								float center = tplist1[z].getf(id);
								for (int dz = -1; dz <= 1; dz++){
									for (int dy = -1; dy <= 1; dy++){
										for (int dx = -1; dx <= 1; dx++){
											if (dx != 0 || dy != 0 || dz != 0)
												ismaxima = (ismaxima && center > tplist1[z+dz].getf(id+dy*imageW+dx));
										}
									}
								}
								tplist2[z].setf(id, ismaxima ? center : 0.0f);
							}
						}					
					}//	for (int i = ai.getAndIncrement(); i < names.length;
				}
			};//threads[ithread] = new Thread() {
		}//	for (int ithread = 0; ithread < threads.length; ithread++)
		startAndJoin(threads);
		}

		IJ.showProgress(2, 5);
		
		IJ.log("Getting Maxima...");
				
		ArrayList<LMaxima> al = new ArrayList<LMaxima>();
		for(int z = zr; z < imageD-zr; z++) {
			for(int y = r; y < imageH-r; y++) {
				for(int x = r; x < imageW-r; x++) {
					float val = tplist2[z].getf(y*imageW+x);
					if (val > 0)
						al.add(new LMaxima(x, y, z, val));
				}
			}
		}
		al.sort((a,b)-> (int)Math.signum(b.score-a.score));
		
		ArrayList<LMaxima> al2 = new ArrayList<LMaxima>();
		for (LMaxima a : al) {
			float val = tplist2[a.z].getf(a.y*imageW + a.x);
			if (val > 0.0f) {
				for (int dz = -2*zr; dz <= 2*zr; dz++){
					for (int dy = -2*rad_; dy <= 2*rad_; dy++){
						for (int dx = -2*rad_; dx <= 2*rad_; dx++){
							int xx = a.x + dx;
							int yy = a.y + dy;
							int zz = a.z + dz;
							if (xx >= 0 && xx < imageW &&
								yy >= 0 && yy < imageH &&
								zz >= 0 && zz < imageD &&
								mask2[Math.abs(dz)][Math.abs(dy)][Math.abs(dx)] == 1) {
								tplist2[zz].setf(yy*imageW + xx, 0.0f);
							}
						}
					}
				}
				al2.add(a);	
			}
		}
		if (al2.size() == 0) {
			IJ.log("not found");
			tmpimp1.show();
			tmpimp2.show();
			newimp.close();
			return;
		}
		float globalmax = al2.get(0).score;
		IJ.log("total number: "+al2.size());

				
		ResultsTable rt = Analyzer.getResultsTable();
		if (rt == null) {
			rt = new ResultsTable();
			Analyzer.setResultsTable(rt);
		}
		else
			rt.reset();

		int bins = 128;
		int [] scorehist = new int[bins];
		Arrays.fill(scorehist, 0);
		for (LMaxima a : al2) {
			int iscore = (int)(a.score/globalmax*bins);
			if (iscore == bins) iscore = bins - 1;
			scorehist[iscore]++;
		}

		boolean st_decr = false;
		boolean st_incr = false;
		int inflection = -1;
		
		for (int i = 2; i < bins-2; i++) {
			double df1 = (scorehist[i] - scorehist[i-2]) / (2.0 * al2.size() / bins);
			double df2 = (scorehist[i+2] - scorehist[i]) / (2.0 * al2.size() / bins);
			double ddf = df2 - df1;

			if (!st_decr && ddf < 0.0)
				st_decr = true;
			if (st_decr && inflection == -1 && ddf > 0.0) {
				inflection = i;
				break;
			}
		}

		int cells = 0;
		for (int i = inflection; i < bins; i++) {
			cells += scorehist[i];
		}
		IJ.log("[inflection point]  score: "+(double)inflection*globalmax/bins+"  number: "+cells);

		st_decr = false;
		st_incr = false;
		double max_curvature = 0.0;
		int max_curvature_point = 0;
		for (int i = 2; i < bins-2; i++) {
			double df = (scorehist[i+1] - scorehist[i-1]) / (2.0 * al2.size() / bins);
			double df1 = (scorehist[i] - scorehist[i-2]) / (2.0 * al2.size() / bins);
			double df2 = (scorehist[i+2] - scorehist[i]) / (2.0 * al2.size() / bins);
			double ddf = df2 - df1;
			double curvature = Math.abs(ddf) / Math.pow(1.0+df*df, 3.0/2.0);

			if (!st_decr && ddf < 0.0)
				st_decr = true;
			if (st_decr && ddf > 0.0)
				st_incr = true;
			
			if (st_incr && max_curvature <= curvature) {
				max_curvature = curvature;
				max_curvature_point = i;
			}
		}
		cells = 0;
		for (int i = max_curvature_point; i < bins; i++) {
			cells += scorehist[i];
		}
		IJ.log("[maximum curvature]  score: "+(double)max_curvature_point*globalmax/bins+"  number: "+cells);

		IJ.showProgress(3, 5);
		
		int count = 1;	
		for (LMaxima a : al2) {
			int x = a.x;
			int y = a.y;
			int z = a.z;
			rt.incrementCounter();
			rt.addValue("id", count);
			rt.addValue("score", a.score);
			rt.addValue("x", x);
			rt.addValue("y", y);
			rt.addValue("z", z);

			for (int dz = -zr; dz <= zr; dz++){
				for (int dy = -rad_; dy <= rad_; dy++){
					for (int dx = -rad_; dx <= rad_; dx++){
						if (mask[Math.abs(dz)][Math.abs(dy)][Math.abs(dx)] == 1) {
							int did = (y+dy)*imageW+x+dx;
							float score = a.score;
							if (score > oplist[z+dz].getf(did))
								oplist[z+dz].setf(did, score);
						}
					}
				}
			}
			count++;
		}
		rt.show("Results");
		
		ImagePlus rimp = new Duplicator().run(imp_, ch_, ch_, 1, imageD, fr_, fr_);
		IJ.run(rimp, "32-bit", "");
		ImageStack rstack = rimp.getStack();
		for(int s = 0; s < imageD; s++) 
			rstack.addSlice(oplist[s]);

		double dispmax = imp_.getDisplayRangeMax();
		double dispmin = imp_.getDisplayRangeMin();
		ImagePlus cimp = HyperStackConverter.toHyperStack(rimp, 2, imageD, 1, "xyzct", "composite");
		LUT[] luts = new LUT[2];
		luts[0] = LUT.createLutFromColor(new Color(255,0,255));
		luts[1] = LUT.createLutFromColor(new Color(0,255,0));
		CompositeImage cmp = (CompositeImage)cimp;
		cmp.setTitle(imp_.getTitle()+"_EVE_rad_"+String.valueOf(rad_)+"_th_"+String.valueOf(th_));
		cmp.setLuts(luts);
		cmp.setC(1);
		cmp.setDisplayRange(dispmin, dispmax);
		cmp.setC(2);
		cmp.setDisplayRange((double)max_curvature_point*globalmax/bins, (double)max_curvature_point*globalmax/bins);
		cmp.show();

		IJ.showProgress(4, 5);

		newimp.close();
		rimp.close();
		tmpimp1.close();
		tmpimp2.close();
	} //public void run(ImageProcessor ip) {
	
	/** Create a Thread[] array as large as the number of processors available.
		* From Stephan Preibisch's Multithreading.java class. See:
		* http://repo.or.cz/w/trakem2.git?a=blob;f=mpi/fruitfly/general/MultiThreading.java;hb=HEAD
		*/
	private Thread[] newThreadArray() {
		int n_cpus = Runtime.getRuntime().availableProcessors();
		if (n_cpus > thread_num_) n_cpus = thread_num_;
		if (n_cpus <= 0) n_cpus = 1;
		return new Thread[n_cpus];
	}
	
	/** Start all given threads and wait on each of them until all are done.
		* From Stephan Preibisch's Multithreading.java class. See:
		* http://repo.or.cz/w/trakem2.git?a=blob;f=mpi/fruitfly/general/MultiThreading.java;hb=HEAD
		*/
	public static void startAndJoin(Thread[] threads)
	{
		for (int ithread = 0; ithread < threads.length; ++ithread)
		{
			threads[ithread].setPriority(Thread.NORM_PRIORITY);
			threads[ithread].start();
		}
		
		try
		{   
			for (int ithread = 0; ithread < threads.length; ++ithread)
			threads[ithread].join();
		} catch (InterruptedException ie)
		{
			throw new RuntimeException(ie);
		}
	}
}
