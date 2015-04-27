package main.java;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.apache.spark.api.java.function.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;


public class ParseJSON {

	public static int sizex = 64, sizey = 64;
	public static Mat img; 
	public static int numRows ;
    public static int numCols ;
    
	public static void main (String[] args) throws IOException, ParseException {
		
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		img = Highgui.imread("/home/isinsu/Desktop/DeepManuscripts/preprocessing/IS5337_2_010_00093.tif");
		String filePath = "/home/isinsu/Desktop/DeepManuscripts/preprocessing/IS5337_2_010_00093.json";
		Imgproc.cvtColor(img,img, Imgproc.COLOR_BGR2GRAY);
		numRows = img.rows();
	    numCols = img.cols();
		processLines(img, filePath, "--local");
	}
	
	public static void processLines(Mat imgMat, String filePath, String arg)
	{
		SparkConf conf;
		JavaSparkContext sc;
		if(arg.equals("--local")) {
		    conf = new SparkConf().setAppName("DeepManuscript candidate patches").setMaster("local");
		    sc = new JavaSparkContext(conf);
		}else {
		    conf = new SparkConf().setAppName("DeepManuscript candidate patches");
		    sc = new JavaSparkContext(conf);
		   
		}
		
		 ArrayList<Integer[][]> lines = parseJSON(filePath);
		// System.out.println("Line number: "+ lines.size());
		 JavaRDD<Integer[][]> textLines = sc.parallelize(lines);
		 JavaRDD<List<Vector>> patches = textLines.map(new Function<Integer[][], List<Vector>> (){
			@Override
			public List<Vector> call(Integer[][] arg0) throws Exception {
				List<Vector> result = extractPatches(arg0);
				return result;
			}
			
		 });
		
		patches.saveAsTextFile("/home/isinsu/Desktop/DeepManuscripts/preprocessing/output");
		sc.close();
//		ArrayList<Integer[][]> lines = parseJSON(filePath);
//		System.out.println("Line number: "+ lines.size());
//		for(int i = 0; i < lines.size(); i++)
//		{
//			List<Vector> res =  extractPatches(lines.get(i));
//			System.out.println(res.size());
//		}
		
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
	
	public static List<Vector> extractPatches(Integer[][] boundary) {

		List<Vector> results = new ArrayList<>();
		int stepSize;
		if(sizex > sizey)
			stepSize = sizey/2;
		else
			stepSize = sizex/2;
		
		sortArray(boundary,0);
		int xmin = boundary[0][0];
		int xmax = boundary[boundary.length-1][0];
		sortArray(boundary,1);
		int ymin = boundary[0][1];
		int ymax = boundary[boundary.length-1][1];
		//System.out.println(xmin + " " + xmax + " " + ymin + " " + ymax);
		
		for(int i = xmin; i <= xmax ; i = i+stepSize)
		{
			for(int j = ymin; j <= ymax ; j= j+stepSize)
			{
				double[] patchData = new double[sizex*sizey];
				byte[] lineData = new byte[numCols];
				Mat pat = new Mat();
				if((i-stepSize+1 >= 0) && (i+stepSize+1 <= numRows) && (j-stepSize+1 >= 0) && (j+stepSize+1 <= numCols))
				{
					if(stepSize % 2 == 0)
					{
						pat = img.submat(i-stepSize+1, i+stepSize+1, j-stepSize+1, j+stepSize+1);
						//System.out.println(pat.size().height);
					}
					else
					{
						pat = img.submat(i-stepSize, i+stepSize+1, j-stepSize, j+stepSize+1);
					}
					//Highgui.imwrite("/home/isinsu/Desktop/DeepManuscripts/preprocessing/pat.tif", pat);
					
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

	private static ArrayList<Integer[][]> parseJSON(String filePath){
		
		ArrayList<Integer[][]> textLines = new ArrayList<Integer[][]>();
		
		try {
			// read the json file (full page)
			FileReader reader = new FileReader(filePath);
			
			JSONParser jsonP = new JSONParser();
			JSONObject page = new JSONObject();
			try {
				page = (JSONObject) jsonP.parse(reader);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ParseException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			//get the page_id			
			String page_id = (String) page.get("id");
			//System.out.println("page_id  = " + page_id);
			
			//get the page height and width
			long height = (long) page.get("height");
			long width = (long) page.get("width");
			//System.out.println("height = " + height + "  width = " + width);
			
			//get the number of lines
			JSONArray lines = (JSONArray) page.get("lines");
			int num_lines = lines.size();
			//System.out.println("This page has " + num_lines + " lines");
			
			//nest into lines
			Iterator l = lines.iterator();
			
			while (l.hasNext()) {
				JSONObject line = (JSONObject) l.next();
				
				//get the line_id
				long line_id = (long) line.get("id");
				//System.out.println("line_id =  " + line_id);
				
				//get the x,y,w,h for the line
				long x = (long) line.get("x");
				long y = (long) line.get("y");
				long w = (long) line.get("w");
				long h = (long) line.get("h");
				
				//get the boundaries for line
				String boundaries = (String) line.get("boundaries");
				//System.out.println("boundaries =  " + boundaries);
				
				Integer[][] g_b = get_boundaries(boundaries);
				textLines.add(g_b);
				//System.out.println("First two boundary points are (" + textLines.get(textLines.size()-1)[0][0] + "," + textLines.get(textLines.size()-1)[0][1] + ") and (" + textLines.get(textLines.size()-1)[1][0] + "," + textLines.get(textLines.size()-1)[1][1] + ")");
				
				//get the number of words in that line
				JSONArray words = (JSONArray) line.get("segments");
				int num_words = words.size();
				//System.out.println("This line has " + num_words + " words");
				
				//nest into words				
				Iterator wo = words.iterator();
				
				while (wo.hasNext()) {
					JSONObject word = (JSONObject) wo.next();
					
					//get the word_id
					long word_id = (long) word.get("id");
					//System.out.println("word_id =  " + word_id);
					
					//get the x,y,w,h for the word
					long xw = (long) word.get("x");
					long yw = (long) word.get("y");
					long ww = (long) word.get("w");
					long hw = (long) word.get("h");
					
					//get the boundaries for word
					String boundariesw = (String) word.get("boundaries");
					//System.out.println("boundaries =  " + boundaries);
					
					Integer[][] g_b_w = get_boundaries(boundariesw); 
					//System.out.println("First two boundary points are (" + g_b_w[0][0] + "," + g_b_w[0][1] + ") and (" + g_b_w[1][0] + "," + g_b_w[1][1] + ")");
				}
				
			}
			
		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		}
		return textLines;
		
	}

	private  static Integer[][] get_boundaries(String boundaries) {
		// gets the two dimensional array of the form [size][2] where size is the number 
		// of boundary points and 2 corresponds to the two indices x,y ..
		// "boundaries" string is in the form ((a,b),(c,d),(e,f),.......,(y,z))
		Integer[][] gb;
		String delim = "[(),]+";
		String[] tokens = boundaries.split(delim);
		
		//get total number of numbers in the boundaries which 2 times the actual number of points --> (x,y)
		int size = tokens.length - 1;
		// -1 because 1st entry is ""
		gb = new Integer[size/2][2];
		int i = 0;
		while (i < size) {
			int boundary_index_x = i/2;
			int boundary_index_y = i%2; 
			gb[boundary_index_x][boundary_index_y] = Integer.parseInt(tokens[i+1].trim()); 			
			//System.out.println("token =  " + tokens[i+1] );
			i++;
		}
		
		return gb;
	}
			
}
