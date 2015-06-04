package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 * All Learners (KMeans, Autoencoders,..) should implement this interface.
 * 
 * @author Arttu Voutilainen
 *
 */
public interface Learner extends Function<JavaRDD<Tuple2<Vector, Vector>>,Vector[]> {

	/**
	 * Main method that runs a specific learning procedure. 
	 * Initially, two methods will be implemented:
	 * 	1. K-means
	 *	2. Autoencoder
	 *
	 * @param pairData An RDD that represents the original data from which we learn filters.
	 * @return An array of Vectors that represent the learned filters.
	**/
	public Vector[] call(JavaRDD<Tuple2<Vector, Vector>> pairData);
	
	
	/**
	 * Method that loads the learned features from files.
	 *  
	 * @param filename File that contains the learned features.
	 * @param sc Spark context variable.
	 * @return Array of learned features.
	 **/
	public Vector[] loadFromFile(String filename, JavaSparkContext sc);
	
	
	/**
	 * Saves the learned features.
	 * 
	 * @param features Array of features to save.
	 * @param filename File name.
	 * @param sc Spark context variable.
	 **/
	public void saveToFile(Vector[] features, String filename, JavaSparkContext sc);

}
