package main.java;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;

import scala.collection.Traversable;

public class CoordinateSpectralClustering {
	Vector[] input;
	int type;
	CoordinateMatrix kNN;
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
	public CoordinateSpectralClustering(Vector[] input) {
		this.input = input;
	}

	/**
	 * Getter for vectorsClusters.
	 * 
	 * @return A JavaRDD of Integer containing the IDs of the clusters
	 *         corresponding to each input Vector.
	 */
	public JavaRDD<Integer> getVectorsClusters() {
		return vectorsClusters;
	}

	/**
	 * Train the Spectral Clustering Algorithm using the data provided by the
	 * Input Matrix and return.
	 * 
	 * @param k
	 *            Number of clusters.
	 */
	public void computeClustering(int k) {
		CoordinateMatrix d = getDegreeMatrix(kNN);
		CoordinateMatrix l = computeUnnormalizedLaplacian(kNN, d);
		CoordinateMatrix u = eigenvectorComputing(l, k);
		kmeansTraining(u, k);
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
		KNearestNeighbor kNN = new KNearestNeighbor(input, k, matrixType,
				neighborType, sigma);

		this.kNN = kNN.getWeightedCoordinateMatrix();

	}

	/**
	 * Train KMeansModel and cluster the values.
	 * 
	 * @param y
	 *            Matrix of eingenvectors.
	 * @param k
	 *            Number of clusters.
	 */
	private void kmeansTraining(CoordinateMatrix y, int k) {
		int nbRow = (int) y.numRows();
		int nbCol = (int) y.numCols();
		double[][] yArray = getValues2D(y);
		Vector[] vectors = new Vector[nbRow];

		for (int i = 0; i < nbRow; i++) {
			double[] temp = new double[nbCol];
			for (int j = 0; j < nbCol; j++) {
				temp[j] = yArray[i][j];
			}
			vectors[i] = Vectors.dense(temp);
		}
		List<Vector> data = Arrays.asList(vectors);

		SparkConf conf = new SparkConf()
				.setAppName("K-means in Spectral Clustering");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<Vector> distData = sc.parallelize(data);

		int numIterations = 20;
		KMeansModel clusters = KMeans.train(JavaRDD.toRDD(distData), k,
				numIterations);
		sc.close();
		this.training = clusters;
		this.vectorsClusters = clusters.predict(distData);
	}

	/**
	 * Compute the k eigenvectors of the matrix l.
	 * 
	 * @param l
	 *            Laplacian matrix.
	 * 
	 * @param k
	 *            Number of clusters
	 * @return The k eigenvectors of the matrix l.
	 */
	private CoordinateMatrix eigenvectorComputing(CoordinateMatrix l, int k) {
		int nbRow = (int) l.numRows();
		Array2DRowRealMatrix L = toArray2DRowRealMatrix(l);
		EigenDecomposition eigen = new EigenDecomposition(L);
		double[][] eigenvectorsValues = new double[k][nbRow];
		for (int i = 0; i < k; i++) {
			eigenvectorsValues[i] = eigen.getEigenvector(i).toArray();
		}
		return toMatrix(k, nbRow, eigenvectorsValues);
	}

	/**
	 * Compute the unnormalized Laplacian
	 * 
	 * @param kNN2
	 *            Weighted adjacency matrix W.
	 * @param d
	 *            Degree matrix D.
	 * @return Unnormalized graph Laplacian matrix L
	 */
	private CoordinateMatrix computeUnnormalizedLaplacian(
			CoordinateMatrix kNN2, CoordinateMatrix d) {
		int nbRow = (int) kNN2.numRows();
		int nbCol = (int) kNN2.numCols();
		Array2DRowRealMatrix W = toArray2DRowRealMatrix(kNN2);
		Array2DRowRealMatrix L = toArray2DRowRealMatrix(d);
		L.subtract(W);
		return toMatrix(nbRow, nbCol, L.getData());
	}

	private Array2DRowRealMatrix toArray2DRowRealMatrix(CoordinateMatrix kNN2) {
		return new Array2DRowRealMatrix(getValues2D(kNN2));
	}

	/**
	 * Calculate the degree matrix D from the input matrix W.
	 * 
	 * @param kNN2
	 *            The input Matrix
	 * @return Degree matrix D.
	 */
	private CoordinateMatrix getDegreeMatrix(CoordinateMatrix kNN2) {
		int nbRow = (int) kNN2.numRows();
		int nbCol = (int) kNN2.numCols();
		double[][] values = getValues2D(kNN2);
		double[][] d = new double[nbRow][nbCol];
		for (int i = 0; i < nbRow; i++) {
			for (int j = 0; j < nbCol; j++) {
				d[i][i] += values[i][j];
			}
		}
		return toMatrix(nbRow, nbCol, d);
	}

	/**
	 * Transform an array of arrays into a Matrix
	 * 
	 * @param nbRow
	 *            Number of rows
	 * @param nbCol
	 *            Number of columns
	 * @param d
	 *            Array of arrays of doubles
	 * @return The Matrix
	 */
	private CoordinateMatrix toMatrix(int nbRow, int nbCol, double[][] d) {
		double[] result = new double[nbRow * nbCol];
		for (int i = 0; i < nbRow; i++) {
			for (int j = 0; j < nbCol; j++) {
				result[i + j * nbRow] = d[i][j];
			}
		}
		return coordinateMatrix(nbRow,nbCol,result);
	}

	/**
	 * Create a CoordinateMatrix from an array of doubles.
	 * @param m Number of row
	 * @param n Number of column
	 * @param result Array of doubles.
	 * @return A CoordinateMatrix
	 */
	private CoordinateMatrix coordinateMatrix(int m,int n, double[] result) {
		MatrixEntry[] array = new MatrixEntry[m*n];
		for(int i=0; i<m*n; i++){
			array[i]= new MatrixEntry(i%m,(int)i/n,result[i]);
		}
		
		List<MatrixEntry> data = Arrays.asList(array);

		SparkConf conf = new SparkConf()
				.setAppName("CoordinateMatrix for kNN in Spectral Clustering");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<MatrixEntry> distData = sc.parallelize(data);
		sc.close();
		return new CoordinateMatrix(distData.rdd());
	}

	/**
	 * Transform a matrix to an array of arrays of doubles
	 * 
	 * @param w
	 *            The Matrix to transform
	 * @return An array of arrays of doubles
	 */
	private double[][] getValues2D(CoordinateMatrix w) {
		int nbRow = (int) w.numRows();
		int nbCol = (int) w.numCols();
		double[] values = new double[nbRow * nbCol];
		scala.collection.immutable.List<MatrixEntry> list = w.entries()
				.toLocalIterator().toList();
		values = listToArray(list, nbRow * nbCol);
		double[][] sorted = new double[nbRow][nbCol];
		for (int i = 0; i < nbRow * nbCol; i++) {
			sorted[i % nbRow][(int) (i / nbCol)] = values[i];
		}
		return sorted;
	}

	private double[] listToArray(Traversable<MatrixEntry> traversable, int i) {
		double[] result = new double[i];
		result[0] = traversable.head().value();
		double[] temp;
		if (i > 0) {
			temp = listToArray(traversable.tail(), i - 1);
			for (int j = 0; j < i - 1; j++) {
				result[j + 1] = temp[j];
			}
		}
		return result;
	}

}
