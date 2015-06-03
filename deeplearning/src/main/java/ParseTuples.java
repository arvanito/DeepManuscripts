package main.java;

import org.apache.commons.lang.StringUtils;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

/**
 * Parse input files to Vectors. From https://github.com/arvanito/test_spark/blob/master/src/main/java/PreProcess.java
 * Implements Function, so that this can be used inside JavaRDD::map.
 */
public class ParseTuples implements Function<String, Tuple2<Vector, Vector>> {

	private static final long serialVersionUID = -5762727282965079666L;

	/**
	 * Method that parses tuples of the form <Vector, Vector>.
	 * 
	 * @param line Input string that represents one Tuple2<Vector, Vector>.
	 * @return Tuple2<Vector, Vector> data.
	 */
	@Override
	public Tuple2<Vector, Vector> call(String line) throws Exception {
		
		// remove the parentheses from the start and the end of the string, CHECK THIS!!!
		line = line.substring(1, line.length());
		
		// find the first part of the tuple
		// it should be contained in the first square brackets
		int firstIndexOpen = line.indexOf('[');
		int firstIndexClose = line.indexOf(']');
		String firstTuple = line.substring(firstIndexOpen+1, firstIndexClose);
		
		// find the second part of the tuple 
		// it should be contained in the second square brackets
		int secondIndexOpen = StringUtils.ordinalIndexOf(line, "[", 2);
		int secondIndexClose = StringUtils.ordinalIndexOf(line, "]", 2);
		String secondTuple = line.substring(secondIndexOpen+1, secondIndexClose);
		
		// now convert the two strings into arrays of doubles
		
		// first string
		String[] sarrayFirst = firstTuple.split(",");	
		double[] valuesFirst = new double[sarrayFirst.length];
		for (int i = 0; i < sarrayFirst.length; i++) {
			valuesFirst[i] = Double.parseDouble(sarrayFirst[i]);	// creates a double from a string
		}
		
		// second string
		String[] sarraySecond = secondTuple.split(",");	
		double[] valuesSecond = new double[sarraySecond.length];
		for (int i = 0; i < sarraySecond.length; i++) {
			valuesSecond[i] = Double.parseDouble(sarraySecond[i]);	// creates a double from a string
		}
		
		// return the tuple
		return new Tuple2<Vector, Vector>(Vectors.dense(valuesFirst), Vectors.dense(valuesSecond));
	}

}
