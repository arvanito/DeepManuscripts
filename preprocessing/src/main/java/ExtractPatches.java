package main.java;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.json.JSONException;
import org.json.JSONObject;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import scala.Tuple2;



public class ExtractPatches {
	
	public static int sizex = 64, sizey = 64;
	public static int nPatch = 10;
	
	/**
	processLines takes as input the image of the page which is of type Mat and the location of the JSON file. It parallelizes the patch extraction operation for each text line in the corresponding page. It saves the result patches as JavaRDD of Tuple2 of Vectors in a text file.
	**/
	
	public static List<Tuple2<Vector,Vector>> processLines(String fname, Tuple2<ImageData, String> tuple2)
	{
		ImageData imgdata = tuple2._1();
		JSONObject page = null;
		try {
			page = new JSONObject(tuple2._2());
		} catch (JSONException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 ArrayList<Tuple2<Integer[],Integer[][]>> lines = ParseJSON.parseJSON(page);
		 Mat img = imgdata.getImage();
		 int numRows = img.rows();
		 int numCols = img.cols();
		 List<Tuple2<Vector,Vector>> result = new ArrayList<Tuple2<Vector,Vector>>();
		 for(int i = 0; i < lines.size(); i++)
		 {
			 result.addAll(extractPatches2(img, numRows, numCols,lines.get(i)));
		 }
		return result;
		 
		 
		//patches.saveAsObjectFile("/home/isinsu/Desktop/DeepManuscripts/preprocessing/output");
		
	}
	
	public static void sortArray(Integer myArray[][], final int dim)
	{ 
	    Arrays.sort(myArray, new Comparator<Integer[]>()
	    { 
	            @Override 
	            public int compare(Integer[] o1, Integer[] o2) 
	            {
	                    return o1[dim].compareTo(o2[dim]); 
	            } 
	    }); 
	}
	
	
	/**
	extractPathces takes as input the two dimensional array of boundary coordinates and it uses sliding window approach within the boundaries and eliminates blank patches. The output includes a tuple of two vectors where the first vector contains
	- line id
	- (x,y) coordinates of upper left corner of the patch
	- width of the patch
	- height of the patch
	and the second vector containes the column major vectorized patches.
	Similar approaches are used in extractPatches2 and extractPatches3
	**/
	
	public static List<Tuple2<Vector,Vector>> extractPatches(Mat img, int numRows, int numCols, Tuple2<Integer[],Integer[][]> boundary) {

		List<Tuple2<Vector,Vector>> results = new ArrayList<>();
		int stepSize;
		int stepSizex;
		int stepSizey;
		//if(sizex > sizey)
			stepSizex = sizex/2;
		//elsey
			stepSizey = sizey;
		
		sortArray(boundary._2,1);
		int xmin = boundary._2[0][1];
		int xmax = boundary._2[boundary._2.length-1][1];
		sortArray(boundary._2,0);
		int ymin = boundary._2[0][0];
		int ymax = boundary._2[boundary._2.length-1][0];
		//System.out.println(xmin + " " + xmax + " " + ymin + " " + ymax);
		
		for(int i = xmin; i <= xmax ; i = i+stepSizex)
		{
			for(int j = ymin; j <= ymax ; j= j+stepSizey)
			{
				//System.out.println("ymin: "+ ymin + "ymax: "+ ymax + "current y: " + j);
				double[] patchData = new double[sizex*sizey];
				byte[] lineData = new byte[numCols];
				Mat pat = new Mat();
				if((i-stepSizex+1 >= 0) && (i+stepSizex+1 <= numRows) && (j-stepSizey+1 >= 0) && (j+stepSizey+1 <= numCols))
				{
					if(stepSizey % 2 == 0)
					{
						pat = img.submat(i-stepSizex+1, i+stepSizex+1, j-stepSizey/2+1, j+stepSizey/2+1);
						//System.out.println(pat.size().height);
					}
					else
					{
						//pat = img.submat(i-stepSize, i+stepSize+1, j-stepSize, j+stepSize+1);
					}
					
					
					/*Standard Deviation:
					 * */
					double total = 0;
					for(int a = 0; a < pat.rows(); a++)
					{
						for (int b = 0; b < pat.cols(); b++)
						{
							byte [] pat_data = new byte[1];
							pat.get(a, b, pat_data);
							double pat_data_val = (pat_data[0]>= 0 ? pat_data[0] : 256+pat_data[0]);
							total += pat_data_val;
							
						}
					}
					double mean = total/(pat.cols()*pat.rows());
					
					double var = 0;
					
					for(int a = 0; a < pat.rows(); a++)
					{
						for (int b = 0; b < pat.cols(); b++)
						{
							byte [] pat_data = new byte[1];
							pat.get(a, b, pat_data);
							double pat_data_val = (pat_data[0]>= 0 ? pat_data[0] : 256+pat_data[0]);
							var += ((pat_data_val - mean) * (pat_data_val - mean));
							
						}
					}
					var /= (pat.rows()*pat.cols());
					double std_dev = Math.sqrt(var);
					//System.out.println("Standard Deviation: " + std_dev );
					if(std_dev > 15)
					{
						//Highgui.imwrite("/home/ashish/Desktop/BDProject/output/pat.tif", pat);
						Mat temppatch = pat.clone();
						Mat patch = temppatch.t();
						patch = patch.reshape(0, 1);
						//System.out.println(patch.rows() + " " + patch.cols());
						for(int k = 0; k < patch.cols(); k++)
						{
							byte [] data = new byte[1];
							patch.get(0, k, data);
							patchData[k] = (data[0]>= 0 ? data[0] : 256+data[0]);
						}
						double [] coord = new double[6];
						coord[0] = boundary._1[0];
						coord[1] = boundary._1[1];
						coord[2] = i-(sizex/2);
						coord[3] = j-(sizey/2);
						coord[4] = sizex;
						coord[5] = sizey;
						results.add(new Tuple2<Vector, Vector>(new DenseVector(coord),new DenseVector(patchData)));
						 
					}
					
				}
				
			}
		}
		
		return results;
	}
	
	
	public static List<Tuple2<Vector,Vector>> extractPatches2(Mat img, int numRows, int numCols,Tuple2<Integer[],Integer[][]>  boundary) {
		List<Tuple2<Vector,Vector>> results = new ArrayList<Tuple2<Vector,Vector>>();
		int patchSize;
		int stepSizex;
		int stepSizey;
		stepSizex = sizex/2;
		stepSizey = sizey;
		if(sizex > sizey)
			patchSize = sizey/2;
		else
			patchSize = sizex/2;
		
		//sortArray(boundary,1);
		//int xmin = boundary[0][1];
		//System.out.println("xmin: "+ xmin);
		//int xmax = boundary[boundary.length-1][1];
		sortArray(boundary._2,1);
		int xmin = boundary._2[0][1];
		int xmax = boundary._2[boundary._2.length-1][1];
		//sortArray(boundary._2,0);
		//int ymin = boundary._2[0][0];
		//int ymax = boundary._2[boundary._2.length-1][0];
		
		Integer[]tempboundary = new Integer[boundary._2.length];
		Integer[]tempboundary2= new Integer[boundary._2.length];
		for(int s = 0; s < boundary._2.length; s++)
		{
			tempboundary[s] = boundary._2[s][1];
		}
		int stepSize = patchSize;
		for(int i = xmin; i <= xmax+stepSizex/2 ; i = i+stepSizex)
		{
			//System.out.println("x: "+i);
			ArrayList<Integer> yval = new ArrayList<Integer>();
			int ind = 0;
			int val =-1;
			int ival = i;
			for(int l=0;l<tempboundary.length;l++){
				if(tempboundary[l]==ival){
					val=l;
					break;
				}
			}
		    tempboundary2 = Arrays.copyOfRange(tempboundary, 0, tempboundary.length);
		    int l=0;
		    int range = 2*stepSizex;
		    if(i==xmin+stepSizex/2){range = 2*stepSizex; ival = i-stepSizex/2;}
		    if(i>xmin+stepSizex/2){range = stepSizex; ival = i-stepSizex;}
		    //System.out.println("val:"+val);
			while(ival<i+range)
			{
				if(val!=-1)
				{yval.add(boundary._2[val+ind][0]);}
				tempboundary2 = Arrays.copyOfRange(tempboundary2, val+1, tempboundary2.length);
				//System.out.println("uzunluk" + tempboundary2.length);
				ind += val+1;
				val=-1;
				for(l=0;l<tempboundary2.length;l++){
					if(tempboundary2[l]==ival){
						val=l;
						break;
					}
				}
				if(val==-1)
					{ival++;}
			}
			//System.out.println("yval.size:"+yval.size());
			if(yval.size() >= 2)
			{
				int ymin = Collections.min(yval);
				int ymax = Collections.max(yval);
				//System.out.println("ymin: "+ ymin + " ymax: " + ymax);
				for(int j = ymin; j <= ymax; j= j+stepSizey)
				{
					double[] patchData = new double[sizex*sizey];
					byte[] lineData = new byte[numCols];
					Mat pat = new Mat();
					if((i-patchSize+1 >= 0) && (i+patchSize+1 <= numRows) && (j-(2*patchSize)+1 >= 0) && (j+(2*patchSize)+1 <= numCols))
					{
						if(stepSize % 2 == 0)
						{
							pat = img.submat(i-patchSize+1, i+patchSize+1, j-patchSize+1, j+patchSize+1);
							//System.out.println(pat.size().height);
						}
						else
						{
							pat = img.submat(i-patchSize, i+patchSize+1, j-patchSize, j+patchSize+1);
						}
						double total = 0;
						for(int a = 0; a < pat.rows(); a++)
						{
							for (int b = 0; b < pat.cols(); b++)
							{
								byte [] pat_data = new byte[1];
								pat.get(a, b, pat_data);
								double pat_data_val = (pat_data[0]>= 0 ? pat_data[0] : 256+pat_data[0]);
								total += pat_data_val;
								
							}
						}
						double mean = total/(pat.cols()*pat.rows());
						
						double var = 0;
						
						for(int a = 0; a < pat.rows(); a++)
						{
							for (int b = 0; b < pat.cols(); b++)
							{
								byte [] pat_data = new byte[1];
								pat.get(a, b, pat_data);
								double pat_data_val = (pat_data[0]>= 0 ? pat_data[0] : 256+pat_data[0]);
								var += ((pat_data_val - mean) * (pat_data_val - mean));
								
							}
						}
						var /= (pat.rows()*pat.cols());
						double std_dev = Math.sqrt(var);
						if(std_dev > 15)
						{
						//Highgui.imwrite("/home/arun/Big Data Project/DeepManuscript/preprocessing/pat/" +saveindex+".tif", pat);
						//saveindex = saveindex + 1;
						Mat temppatch = pat.clone();
						Mat patch = temppatch.t();
						patch = patch.reshape(0, 1);
						//System.out.println(patch.rows() + " " + patch.cols());
						for(int k = 0; k < patch.cols(); k++)
						{
							byte [] data = new byte[1];
							patch.get(0, k, data);
							patchData[k] = (data[0]>= 0 ? data[0] : 256+data[0]);
						}
						double [] coord = new double[6];
						coord[0] = boundary._1[0];
						coord[1] = boundary._1[1];
						coord[2] = i-(sizex/2);
						coord[3] = j-(sizey/2);
						coord[4] = sizex;
						coord[5] = sizey;
		
						results.add(new Tuple2<Vector, Vector>(new DenseVector(coord),new DenseVector(patchData))); 
						}
					}
				}
			}
		}
		
		return results;
	}
	
	public static List<Vector> extractPatches3(Mat img, int numRows, int numCols, Integer[][] boundary) {
		List<Vector> results = new ArrayList<>();
		int stepSize;
		if(sizex > sizey)
			stepSize = sizey/2;
		else
			stepSize = sizex/2;
		
		sortArray(boundary,1);
		int xmin = boundary[0][1];
		//System.out.println("xmin: "+ xmin);
		int xmax = boundary[boundary.length-1][1];
		int rangex = xmax - xmin;
		Integer[]tempboundary = new Integer[boundary.length];
		Integer[]tempboundary2= new Integer[boundary.length];
		for(int s = 0; s < boundary.length; s++)
		{
			tempboundary[s] = boundary[s][0];
		}
		for(int i = 0; i < nPatch ; )
		{
			int randx = (int)(Math.random()*rangex + xmin);
			ArrayList<Integer> yval = new ArrayList<Integer>();
			int ind = 0;
			int val = Arrays.binarySearch(tempboundary, i);
		    tempboundary2 = Arrays.copyOfRange(tempboundary, 0, tempboundary.length);
			while(val >= 0)
			{
				yval.add(boundary[val+ind][0]);
				tempboundary2 = Arrays.copyOfRange(tempboundary2, val+1, tempboundary2.length);
				//System.out.println("uzunluk" + tempboundary2.length);
				ind += val+1;
				val = Arrays.binarySearch(tempboundary2, i);
			}
			if(yval.size() != 0)
			{
				int ymin = Collections.min(yval);
				int ymax = Collections.max(yval);
				int rangey = ymax - ymin;
				int randy = (int)(Math.random()*rangey + ymin);
				double[] patchData = new double[sizex*sizey];
				byte[] lineData = new byte[numCols];
				Mat pat = new Mat();
				if((randx-stepSize+1 >= 0) && (randx+stepSize+1 <= numRows) && (randy-stepSize+1 >= 0) && (randy+stepSize+1 <= numCols))
				{
					if(stepSize % 2 == 0)
					{
						pat = img.submat(randx-stepSize+1, randx+stepSize+1, randy-stepSize+1, randy+stepSize+1);
						//System.out.println(pat.size().height);
					}
					else
					{
						pat = img.submat(randx-stepSize, randx+stepSize+1, randy-stepSize, randy+stepSize+1);
					}
					i = i+1; //We extracted a random patch
					//Highgui.imwrite("/home/ashish/Desktop/BDProject/output/p" +saveindex+".tif", pat);
					Mat temppatch = pat.clone();
					Mat patch = temppatch.t();
					patch = patch.reshape(0, 1);
					//System.out.println(patch.rows() + " " + patch.cols());
					for(int k = 0; k < patch.cols(); k++)
					{
						byte [] data = new byte[1];
						patch.get(0, k, data);
						patchData[k] = (data[0]>= 0 ? data[0] : 256+data[0]);
					}
	
					results.add(new DenseVector(patchData)); 
					
				}
				
			}
			
		}
		return results;
	}
	


}
