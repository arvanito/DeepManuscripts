package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

/**
 * 
 * Interface for a deep learning layer. This allows a main loop which just calls each of the functions for each layer.
 *
 */
public interface DeepLearningLayer {
	/**
	 * Takes in data, does for example K-means clustering and returns the features (like cluster centroids).
	 * 
	 * @param data An RDD containing the data on which learning is to happen
	 * @return An RDD consisting of the learned features as Vectors
	 */
	public JavaRDD<Vector> learnFeatures(JavaRDD<Vector> data);
	
	/**
	 * Extracts a given set of features from a given set of data. Returns one Vector per input data Vector,
	 * so for example convolution will have to reshape the resulting matrices (from one input data Vector) 
	 * into vectors and append them into one big Vector. 
	 * 
	 * @param data An RDD of Vectors, from which the features are to be extracted.
	 * @param features An RDD of Vectors, where each Vector is a feature
	 * @return An RDD of extracted features, where each Vector corresponds to one Vector in data
	 */
	public JavaRDD<Vector> extractFeatures(JavaRDD<Vector> data, JavaRDD<Vector> features);
	
	/**
	 * Pools data to reduce dimensionality. Probably requires some knowledge of the input Vector's structure
	 * (like "this is 1000 times 32x32 matrices"). 
	 * 
	 * @param data An RDD of Vectors, where each Vector represents one element in input dataset (pooling happens only inside a Vector).
	 * @return An RDD of the pooled Vectors
	 */
	public JavaRDD<Vector> pool(JavaRDD<Vector> data);
}