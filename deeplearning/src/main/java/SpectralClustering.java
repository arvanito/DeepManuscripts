package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;

public class SpectralClustering {
	int limit = 50;
	SparseSpectralClustering ssc;
	CoordinateSpectralClustering csc;
	boolean sparse;

	Vector[] input;
	Matrix kNN;
	CoordinateMatrix kNN2;
	KMeansModel training;
	JavaRDD<Integer> vectorsClusters;

	/**
	 * General constructor for Spectral Clustering. <br>
	 * To cluster the input, first use the <i>.computeKNN</i> method and the
	 * <i>.computeClustering</i> method and finally get the result by using
	 * <i>getVectorsClusters</i> method.
	 * 
	 * @param input
	 *            Input array of Vectors to sort by clusters.
	 */
	public SpectralClustering(Vector[] input) {
		this.input = input;
		if (input.length > limit) {
			csc = new CoordinateSpectralClustering(input);
			ssc = null;
			sparse = false;
		} else {
			ssc = new SparseSpectralClustering(input);
			csc = null;
			sparse = true;
		}
	}

	/**
	 * Getter for vectorsClusters.
	 * 
	 * @return A JavaRDD of Integer containing the IDs of the clusters
	 *         corresponding to each input Vector.
	 */
	public JavaRDD<Integer> getVectorsClusters() {
		if (sparse) {
			return ssc.getVectorsClusters();
		} else {
			return csc.getVectorsClusters();
		}
	}

	/**
	 * Train the Spectral Clustering Algorithm using the data provided by the
	 * Input Matrix and return.
	 * 
	 * @param k
	 *            Number of clusters.
	 */
	public void computeClustering(int k) {
		if (sparse) {
			ssc.computeClustering(k);
		} else {
			csc.computeClustering(k);
		}
	}

	/**
	 * Compute the weighted Matrix using <i>k</i>-Nearest Neighbor algorithm.
	 * 
	 * @param k
	 *            Number of nearest neighbors to consider.
	 * @param matrixType
	 *            The type of the result matrix:
	 *            <ul>
	 *            <li>0: for an unweighted matrix.
	 *            <li>1: for a Gaussian similarity function
	 *            </ul>
	 * @param neighborType
	 *            The type of the kNN matrix:
	 *            <ul>
	 *            <li>0: Not symmetric
	 *            <li>1: Mutual
	 *            <li>2: Normal
	 *            </ul>
	 * @param sigma
	 *            The value of sigma for the Gaussian similarity function
	 */
	public void computeKNN(int k, int matrixType, int neighborType, double sigma) {
		if (sparse) {
			ssc.computeKNN(k, matrixType, neighborType, sigma);
		} else {
			csc.computeKNN(k, matrixType, neighborType, sigma);
		}
	}
}
