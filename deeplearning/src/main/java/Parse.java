package main.java;

import java.util.ArrayList;

import org.apache.commons.lang.StringUtils;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Parse input files to Vectors. From https://github.com/arvanito/test_spark/blob/master/src/main/java/PreProcess.java
 * Implements Function, so that this can be used inside JavaRDD::map.
 */
public class Parse implements Function<String, Vector> {

	private static final long serialVersionUID = -5762727282965079666L;
	

	@Override
	public Vector call(String line) throws Exception {
		
		// remove square brackets from the start and the end of the string
		//line = line.substring(1, line.length()-1);
		
		// remove the parentheses from the start and the end of the string
		line = line.trim().substring(1, line.length()-1);
				
		// find the first part of the tuple
		// it should be contained in the first square brackets
		int firstIndexOpen = line.indexOf('[');
		int firstIndexClose = line.indexOf(']');
		

		// find the second part of the tuple 
		// it should be contained in the second square brackets
		int secondIndexOpen = StringUtils.ordinalIndexOf(line, "[", 2);
		int secondIndexClose = StringUtils.ordinalIndexOf(line, "]", 2);
		String secondTuple = line.substring(secondIndexOpen+1, secondIndexClose);

	

		// second string
		String[] sarraySecond = secondTuple.split(",");	
		double[] valuesSecond = new double[sarraySecond.length];
		for (int i = 0; i < sarraySecond.length; i++) {
			valuesSecond[i] = Double.parseDouble(sarraySecond[i]);	// creates a double from a string
		}
		
		return Vectors.dense(valuesSecond);
		
		
	}

}


//@Override
//public Iterable<Vector> call(String line) throws Exception {
//	
//	// remove square brackets from the start and the end of the string
//	//line = line.substring(1, line.length()-1);
//	
//	// remove the parentheses from the start and the end of the string
//	line = line.trim().substring(1, line.length()-1);
//			
//	// find the first part of the tuple
//	// it should be contained in the first square brackets
//	int firstIndexOpen = line.indexOf('[');
//	int firstIndexClose = line.indexOf(']');
//	
//
//	// find the second part of the tuple 
//	// it should be contained in the second square brackets
//	int secondIndexOpen = StringUtils.ordinalIndexOf(line, "[", 2);
//	int secondIndexClose = StringUtils.ordinalIndexOf(line, "]", 2);
//	String secondTuple = line.substring(secondIndexOpen+1, secondIndexClose);
//
//
//
//	// second string
//	String[] sarraySecond = secondTuple.split(",");	
//	if (sarraySecond.length == size){
//	double[] valuesSecond = new double[sarraySecond.length];
//	for (int i = 0; i < sarraySecond.length; i++) {
//		valuesSecond[i] = Double.parseDouble(sarraySecond[i]);	// creates a double from a string
//	}
//	ArrayList<Vector> result = new ArrayList<Vector>();
//	result.add(Vectors.dense(valuesSecond));
//	return result;
//	} else return null;
//	
//	
//}
