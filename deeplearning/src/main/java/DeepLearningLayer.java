package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

/**
 * 
 * Interface for a deep learning layer. This allows a main loop which just calls each of the functions for each layer.
 *
 * @author Arttu Voutilainen
 * 
 */
public interface DeepLearningLayer {
	
	/**
	 * Preprocesses data using ZCA whitening.
	 * 
	 * @param data An RDD containing the input data for preprocessing
	 * @return Preprocessed data
	 */
	public JavaRDD<Vector> preProcess(JavaRDD<Vector> data);
	
	
	/**
	 * Takes in data, does for example K-means clustering and returns the features (like cluster centroids).
	 * 
	 * @param data An RDD containing the data on which learning is to happen
	 * @return An Vector array of the learned features
	 * @throws Exception 
	 */
	public Vector[] learnFeatures(JavaRDD<Vector> data) throws Exception;
	
	
	/**
	 * Extracts a given set of features from a given set of data. Returns one Vector per input data Vector,
	 * so for example convolution will have to reshape the resulting matrices (from one input data Vector) 
	 * into vectors and append them into one big Vector. 
	 * 
	 * @param data An RDD of Vectors, from which the features are to be extracted
	 * @param configLayer Current layer configuration extracted from the protocol buffer description
	 * @param features An RDD of Vectors, where each Vector is a feature
	 * @return An RDD of extracted features, where each Vector corresponds to one Vector in data
	 */
	public JavaRDD<Vector> extractFeatures(JavaRDD<Vector> data, Vector[] features);
	
	
	/**
	 * Pools data to reduce dimensionality. Probably requires some knowledge of the input Vector's structure
	 * (like "this is 1000 times 32x32 matrices"). 
	 * 
	 * @param data An RDD of Vectors, where each Vector represents one element in input dataset (pooling happens only inside a Vector).
	 * @return An RDD of the pooled Vectors
	 */
	public JavaRDD<Vector> pool(JavaRDD<Vector> data);

	
     /**
	 * Execute all of this layer for the given data.
	 *
	 * @param data An RDD containing the original data
	 * @return An RDD containing a new representation of the data
	 * @throws Exception
	 */
	public JavaRDD<Vector> train(JavaRDD<Vector> data1, JavaRDD<Vector> data2, boolean notLast) throws Exception;

	
    /**
	 * Computes the descriptors of the input image patches.
	 *
	 * @param data An RDD of size N x M containing candidate image patches where
	 * 				N is the number of patches and M is the product of the patch sizes (width x height).
	 * @return An RDD containing a new representation of the data, of size N x (descriptor size)
	 * @throws Exception
	 */
	 public JavaRDD<Vector> test(JavaRDD<Vector> data) throws Exception;
	 
	 
	 /**
	  * Sets the SparkContext, it is needed for saving data into files
	  * 
	  * @param sc The SparkContext variable
	  */
	 public void setSparkContext(JavaSparkContext sc);
	 
	 
	 /**
	  * Sets a boolean variable that checks if we save our trained model, or not
	  * 
	  * @param value True of false
	  */
	 public void setSaveModel(boolean value);
	 
	 
	 /**
	  * Gets the boolean variable that denotes if we save the trained model, or not
	  * 
	  * @return True or false
	  */
	 public boolean getSaveModel();
}