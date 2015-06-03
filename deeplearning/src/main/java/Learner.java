package main.java;

import org.apache.spark.api.java.JavaRDD;
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
	 * @param data An RDD that represents the original data from which we learn filters.
	 * @return An array of Vectors that represent the learned filters.
	**/
	public Vector[] call(JavaRDD<Tuple2<Vector, Vector>> data);

}
