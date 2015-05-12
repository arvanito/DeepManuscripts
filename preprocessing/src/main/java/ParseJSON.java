
/**
ParseJSON class includes the methods for parsing the JSON file of a specific image, extracting the boundaries of each text line and applying the patch extraction with a sliding window approach for every line.
processLines: The purpose of this method is to parallelize the patch extraction approach for every text line using flatMap.
extractPatches: There are various versions of this method. The current method of patch extraction uses a sliding window approach considering that redundant patches crossing the text line boundaries are eliminated. Blank patches are also eliminated by examining the variance of patches and discardignt he ones with low variance.
parseJSON: For a given image, we extract its JSON data and parse it in order to obtain the information about page id, page height and width, text line id, text line height and width, text line location and boundary coordinates.
**/
package main.java;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.simple.parser.ParseException;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import scala.Tuple2;


public class ParseJSON {

	public static int sizex = 64, sizey = 64;
	public static Mat img; 
	public static int numRows ;
    	public static int numCols ;
    	public static int saveindex = 0;
    	public static int nPatch = 10;
	public static void main (String[] args) throws IOException, ParseException {
		
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		
	}


/**
parseJSON is a parser function for the corresponding JSON files of pages.
**/
	static ArrayList<Tuple2<Integer[],Integer[][]>> parseJSON(JSONObject page){
		
		ArrayList<Tuple2<Integer[],Integer[][]>> textLines = new ArrayList<Tuple2<Integer[],Integer[][]>>();
		try{
			//get the page_id		
			String page_ids = page.getString("id");
			Integer page_id = Integer.parseInt(page_ids.substring(page_ids.length()-3, page_ids.length())); 
			//System.out.println("page_id  = " + page_id);
			//get the page height and width
			long height = (long) page.getLong("height");
			long width = (long) page.getLong("width");
			//System.out.println("height = " + height + "  width = " + width);
			
			//get the number of lines
			JSONArray lines = (JSONArray) page.get("lines");
			int num_lines = lines.length();
			//System.out.println("This page has " + num_lines + " lines");
			
			//nest into lines
			
			for(int t = 0; t < lines.length(); t++){
				
				JSONObject line = lines.getJSONObject(t);
				
				//get the line_id
				long line_ids = line.getLong("id");
				Integer line_id =  (int)line_ids; 
				//System.out.println("line_id =  " + line_id);
				
				//get the x,y,w,h for the line
				
				long x =  line.getLong("x");
				long y =  line.getLong("y");
				long w =  line.getLong("w");
				long h =  line.getLong("h");
			
				
				
				//get the boundaries for line
				String boundaries = null;
				
				boundaries =  line.getString("boundaries");
				
				//System.out.println("boundaries =  " + boundaries);
				Integer[] ids = new Integer[2];
				ids[0] = page_id;
				ids[1] = line_id;
				Integer[][] g_b = get_boundaries(boundaries);
				textLines.add(new Tuple2<Integer[],Integer[][]>(ids,g_b));
				//System.out.println("First two boundary points are (" + textLines.get(textLines.size()-1)[0][0] + "," + textLines.get(textLines.size()-1)[0][1] + ") and (" + textLines.get(textLines.size()-1)[1][0] + "," + textLines.get(textLines.size()-1)[1][1] + ")");
				
				//get the number of words in that line
				JSONArray words = null;
				
				words = (JSONArray) line.get("segments");
				
				int num_words = words.length();
				//System.out.println("This line has " + num_words + " words");
				
				//nest into words				
				for(int g = 0; g < words.length(); g++){
				
					JSONObject word = words.getJSONObject(g);
					
					//get the word_id
					
					long word_id = word.getLong("id");
					
					//System.out.println("word_id =  " + word_id);
					
					//get the x,y,w,h for the word
					
					long xw =  word.getLong("x");
					long yw =  word.getLong("y");
					long ww =  word.getLong("w");
					long hw =  word.getLong("h");
					
					//get the boundaries for word
					String boundariesw = null;
					
					boundariesw =  word.getString("boundaries");
					
					//System.out.println("boundaries =  " + boundaries);
					
					Integer[][] g_b_w = get_boundaries(boundariesw); 
					//System.out.println("First two boundary points are (" + g_b_w[0][0] + "," + g_b_w[0][1] + ") and (" + g_b_w[1][0] + "," + g_b_w[1][1] + ")");
				}
				
			}
		}catch(JSONException e){
			e.printStackTrace();
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
