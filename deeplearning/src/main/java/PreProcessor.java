package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;


/**
 * All PreProcessors (PCA, ZCA, etc) should implement this interface.
 *  
 * @author Nikolaos Arvanitopoulos
 *
 */
public interface PreProcessor extends Function<Vector, Vector>{

	/**
	 * Main method that runs a preprocessing pipeline on the data points.
	 * 
	 * @param data Input data points in a distributed dataset
	 * @param configLayer Layer configuration
	 * @return preprocessed distributed dataset
	 */
	public JavaRDD<Vector> preprocessData(JavaRDD<Vector> data, ConfigBaseLayer configLayer);

	
	/**
	 * Method that preprocessed data in parallel. 
	 * 
	 * @param data Input data in Vector format
	 * @return Preprocessed output
	 */
	public Vector call(Vector data) throws Exception;
	
}
