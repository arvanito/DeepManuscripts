package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;


/**
 * All PreProcessors (PCA, ZCA, etc) should implement this interface.
 *  
 * @author Nikolaos Arvanitopoulos
 *
 */
public interface PreProcessor extends Function<Vector, Vector>{

	/**
	 * Method that sets the layer configuration.
	 * 
	 * @param configLayer Layer configuration object.
	 */
	public void setConfigLayer(ConfigBaseLayer configLayer);
	
	
	/**
	 * Main method that runs a pre-processing pipeline on the data points.
	 * 
	 * @param data Input data points in a distributed dataset.
	 * @return pre-processed distributed dataset.
	 */
	public JavaRDD<Tuple2<Vector, Vector>> preprocessData(JavaRDD<Tuple2<Vector, Vector>> data);
	
	
	/**
	 * Method that pre-processes data in parallel. 
	 * 
	 * @param data Input data
	 * @return Pre-processed output.
	 */
	public Vector call(Vector data) throws Exception;
	
	
	/**
	 * Method that loads the pre-processing parameters from files.
	 * First file name is the mean vector, second file name is the ZCA matrix.
	 *  
	 * @param filename Array of files that contain pre-processing variables.
	 * @param sc Spark context variable.
	 **/
	public void loadFromFile(String[] filename, JavaSparkContext sc);
	
	
	/**
	 * Saves the fields necessary to reconstruct a preprocessor object. 
	 * First file name is the mean vector, second file name is the ZCA matrix.
	 * 
	 * @param filename File name.
	 * @param sc Spark context variable.
	 **/
	public void saveToFile(String filename, JavaSparkContext sc);
	
}
