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
public interface PreProcessor extends Function<Tuple2<Vector, Vector>, Tuple2<Vector, Vector>>{

	/**
	 * Method that sets the layer configuration.
	 * 
	 * @param configLayer Layer configuration object
	 */
	public void setConfigLayer(ConfigBaseLayer configLayer);
	
	
	/**
	 * Main method that runs a preprocessing pipeline on the data points.
	 * 
	 * @param data Input data points in a distributed dataset
	 * @return preprocessed distributed dataset
	 */
	public JavaRDD<Tuple2<Vector, Vector>> preprocessData(JavaRDD<Tuple2<Vector, Vector>> data);
	
	
	/**
	 * Method that preprocesses data in parallel. 
	 * 
	 * @param data Input data in Vector format
	 * @return Preprocessed output
	 */
	public  Tuple2<Vector, Vector> call(Tuple2<Vector, Vector> data) throws Exception;
	
	
	/**
	 *  Sets up the preprocessor. It loads the saved weights from the disk.
	 * @param filename
	 **/
	public void loadFromFile(String filename, JavaSparkContext sc);
	
	
	/**
	 *  Saves the fields necessary to reconstruct a preprocessor object. 
	 *  Depending on the preprocessor type, more than one file will be saved.
	 * @param filename
	 **/
	public void saveToFile(String filename, JavaSparkContext sc);
}
