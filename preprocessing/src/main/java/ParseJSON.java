package main.java;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;


import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class ParseJSON {

	private static final String filePath = "/home/ashish/Desktop/BD Project/Datasets/adieu_a_beaucoup_de_personnages/json/IS5337_2_010_00093.json";
	
	public static void main (String[] args) throws IOException, ParseException {
		try {
			// read the json file (full page)
			FileReader reader = new FileReader(filePath);
			
			JSONParser jsonP = new JSONParser();
			JSONObject page = (JSONObject) jsonP.parse(reader);
			
			//get the page_id			
			String page_id = (String) page.get("id");
			System.out.println("page_id  = " + page_id);
			
			//get the page height and width
			long height = (long) page.get("height");
			long width = (long) page.get("width");
			System.out.println("height = " + height + "  width = " + width);
			
			//get the number of lines
			JSONArray lines = (JSONArray) page.get("lines");
			int num_lines = lines.size();
			System.out.println("This page has " + num_lines + " lines");
			
			//nest into lines
			Iterator l = lines.iterator();
			
			while (l.hasNext()) {
				JSONObject line = (JSONObject) l.next();
				
				//get the line_id
				long line_id = (long) line.get("id");
				System.out.println("line_id =  " + line_id);
				
				//get the x,y,w,h for the line
				long x = (long) line.get("x");
				long y = (long) line.get("y");
				long w = (long) line.get("w");
				long h = (long) line.get("h");
				
				//get the boundaries for line
				String boundaries = (String) line.get("boundaries");
				//System.out.println("boundaries =  " + boundaries);
				
				int[][] g_b = get_boundaries(boundaries); 
				System.out.println("First two boundary points are (" + g_b[0][0] + "," + g_b[0][1] + ") and (" + g_b[1][0] + "," + g_b[1][1] + ")");
				
				//get the number of words in that line
				JSONArray words = (JSONArray) line.get("segments");
				int num_words = words.size();
				System.out.println("This line has " + num_words + " words");
				
				//nest into words				
				Iterator wo = words.iterator();
				
				while (wo.hasNext()) {
					JSONObject word = (JSONObject) wo.next();
					
					//get the word_id
					long word_id = (long) word.get("id");
					System.out.println("word_id =  " + word_id);
					
					//get the x,y,w,h for the word
					long xw = (long) word.get("x");
					long yw = (long) word.get("y");
					long ww = (long) word.get("w");
					long hw = (long) word.get("h");
					
					//get the boundaries for word
					String boundariesw = (String) word.get("boundaries");
					//System.out.println("boundaries =  " + boundaries);
					
					int[][] g_b_w = get_boundaries(boundariesw); 
					System.out.println("First two boundary points are (" + g_b_w[0][0] + "," + g_b_w[0][1] + ") and (" + g_b_w[1][0] + "," + g_b_w[1][1] + ")");
					
				}
				
			}
			
		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		}
		
	}

	private static int[][] get_boundaries(String boundaries) {
		// gets the two dimensional array of the form [size][2] where size is the number 
		// of boundary points and 2 corresponds to the two indices x,y ..
		// "boundaries" string is in the form ((a,b),(c,d),(e,f),.......,(y,z))
		int[][] gb;
		String delim = "[(),]+";
		String[] tokens = boundaries.split(delim);
		
		//get total number of numbers in the boundaries which 2 times the actual number of points --> (x,y)
		int size = tokens.length - 1;
		// -1 because 1st entry is ""
		gb = new int[size/2][2];
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
