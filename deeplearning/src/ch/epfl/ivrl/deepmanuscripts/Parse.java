package ch.epfl.ivrl.deepmanuscripts;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Class that parses input files to Vectors. 
 */
public class Parse implements Function<String, Vector> {

	private static final long serialVersionUID = -5762727282965079666L;

	
	/**
	 * Method that parses input data in form of vectors.
	 * 
	 * @param line Input string that represents one data point.
	 * @return Vector data.
	 */
	@Override
	public Vector call(String line) throws Exception {
		
		// remove square brackets from the start and the end of the string
		line = line.substring(1, line.length()-1);
		
		// split the string to an array of string by using "," as a delimiter
		String[] sarray = line.split(",");	
		
		// convert this array of strings to an array of doubles
		double[] values = new double[sarray.length];
		for (int i = 0; i < sarray.length; i++) {
			values[i] = Double.parseDouble(sarray[i]);	// creates a double from a string
		}
		
		return Vectors.dense(values);
	}

}
