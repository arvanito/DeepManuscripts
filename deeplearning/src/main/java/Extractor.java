package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 * All Extractors (Matrix multiply, Convolution,..) should implement this interface.
 * 
 * @author Arttu Voutilainen
 *
 */
public interface Extractor extends Function<Tuple2<Vector, Vector>, Tuple2<Vector, Vector>> {
	
	/**
	 * Sets the mean vector and the ZCA matrix, results of PreProcessZCA.
	 * By default those should be null, in which case pre-processing is not used.
	 * 
	 * @param mean Input mean vector.
	 * @param zca Input ZCA matrix.
	 */
	public void setPreProcessZCA(DenseVector mean, DenseMatrix zca);
	
	
	/**
	 * Sets features to be used by this extractor.
	 *
	 * @param features The features this extractor will use.
	 */
	public void setFeatures(Vector[] features);

	
	/**
	 * Sets the epsilon parameter for contrast normalization.
	 * 
	 * @param eps1 Variable for contrast normalization.
	 */
	public void setEps1(double eps1);
	
	
	/**
	 * Main method that performs feature extraction.
	 *
	 * @param data Input distributed dataset for feature extraction.
	 * @return New data representations.
	 */
	public Tuple2<Vector, Vector> call(Tuple2<Vector, Vector> data);

}
