package ch.epfl.ivrl.deepmanuscripts;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 * 
 * Interface for a deep learning layer. This allows a main loop which just calls each of the functions for each layer.
 *
 * @author Arttu Voutilainen
 * 
 */
public interface DeepLearningLayer {
	
	
	/**
	 * Method that performs pre-processing to the input data.
	 * 
	 * @param data Input distributed dataset for preprocessing.
	 * @return Preprocessed data.
	 */
	public JavaRDD<Tuple2<Vector, Vector>> preProcess(JavaRDD<Tuple2<Vector, Vector>> data);
	
	
	/**
	 * Method that performs feature learning.
	 * 
	 * @param data Input distributed dataset.
	 * @return A Vector array of the learned features.
	 */
	public Vector[] learnFeatures(JavaRDD<Tuple2<Vector, Vector>> data);
	
	
	/** 
	 * Method that performs feature extraction. 
	 * 
	 * @param data Input distributed dataset for feature extraction.
	 * @param features Array of learned features.
	 * @return New representations of the input data points.
	 */
	public JavaRDD<Tuple2<Vector, Vector>> extractFeatures(JavaRDD<Tuple2<Vector, Vector>> data, Vector[] features);
	
	
	/**
	 * Method that performs pooling of representations.
	 * 
	 * @param data Input distributed dataset for pooling.
	 * @return Pooled data representations.
	 */
	public JavaRDD<Tuple2<Vector, Vector>> pool(JavaRDD<Tuple2<Vector, Vector>> data);

	
     /**
	 * Main method that performs a complete training for the current layer.
	 *
	 * @param data1 First input distributed dataset.
	 * @param data2 Second input distributed dataset.
	 * @return Output pooled representations for the current layer.
 	 * @throws Exception Standard Exception object.
	 */
	public JavaRDD<Tuple2<Vector, Vector>> train(JavaRDD<Tuple2<Vector, Vector>> data1, JavaRDD<Tuple2<Vector, Vector>> data2) throws Exception;

	
    /**
	 * Main method that performs a complete test pass through the data for the current layer.
	 *
	 * @param data Input distributed dataset to perform testing over.
	 * @return Output pooled representations for the current layer.
	 * @throws Exception Standard Exception object.
	 */
	 public JavaRDD<Tuple2<Vector, Vector>> test(JavaRDD<Tuple2<Vector, Vector>> data) throws Exception;
	 
	 
	 /**
	  * Sets the SparkContext, it is needed for saving data into files.
	  * 
	  * @param sc The SparkContext variable.
	  */
	 public void setSparkContext(JavaSparkContext sc);
	 
	 
	 /**
	  * Sets a boolean variable that checks if we save our trained model, or not.
	  * 
	  * @param value True of false.
	  */
	 public void setSaveModel(boolean value);
	 
	 
	 /**
	  * Gets the boolean variable that denotes if we save the trained model, or not.
	  * 
	  * @return True or false.
	  */
	 public boolean getSaveModel();
	 
	 
	 /**
	  * Sets a boolean variable that checks if we are in the last layer.
	  * 
	  * @param value True of false.
	  */
	 public void setNotLast(boolean value);
	 
}