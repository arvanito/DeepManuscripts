package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;


/**
 * All PreProcessors (PCA, ZCA, etc) should implement this interface.
 *  
 * @author Nikolaos Arvanitopoulos
 *
 */
public interface PreProcessor {

	/**
	 * Main method that runs a preprocessing pipeline on the data points.
	 * 
	 * @param data Input data points in a distributed dataset
	 * @param configLayer Layer configuration
	 * @return preprocessed distributed dataset
	 */
	public JavaRDD<Vector> preprocessData(JavaRDD<Vector> data, ConfigBaseLayer configLayer);
	
}
