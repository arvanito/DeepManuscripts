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
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class SparseSpectralClustering {
	Vector[] input;
	int type;
	Matrix kNN;
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
	public SparseSpectralClustering(Vector[] input) {
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
		Matrix d = getDegreeMatrix(kNN);
		Matrix l = computeUnnormalizedLaplacian(kNN, d);
		Matrix u = eigenvectorComputing(l, k);
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

		this.kNN = kNN.getSparseWeightedMatrix();

	}

	/**
	 * Train KMeansModel and cluster the values.
	 * 
	 * @param y
	 *            Matrix of eingenvectors.
	 * @param k
	 *            Number of clusters.
	 */
	private void kmeansTraining(Matrix y, int k) {
		int nbRow = y.numRows();
		int nbCol = y.numCols();
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
	private Matrix eigenvectorComputing(Matrix l, int k) {
		int nbRow = l.numRows();
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
	 * @param w
	 *            Weighted adjacency matrix W.
	 * @param d
	 *            Degree matrix D.
	 * @return Unnormalized graph Laplacian matrix L
	 */
	private Matrix computeUnnormalizedLaplacian(Matrix w, Matrix d) {
		int nbRow = w.numRows();
		int nbCol = w.numCols();
		Array2DRowRealMatrix W = toArray2DRowRealMatrix(w);
		Array2DRowRealMatrix L = toArray2DRowRealMatrix(d);
		L.subtract(W);
		return toMatrix(nbRow, nbCol, L.getData());
	}

	private Array2DRowRealMatrix toArray2DRowRealMatrix(Matrix w) {
		return new Array2DRowRealMatrix(getValues2D(w));
	}

	/**
	 * Calculate the degree matrix D from the input matrix W.
	 * 
	 * @param w
	 *            The input Matrix
	 * @return Degree matrix D.
	 */
	private Matrix getDegreeMatrix(Matrix w) {
		int nbRow = w.numRows();
		int nbCol = w.numCols();
		double[][] values = getValues2D(w);
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
	private Matrix toMatrix(int nbRow, int nbCol, double[][] d) {
		double[] result = new double[nbRow * nbCol];
		for (int i = 0; i < nbRow; i++) {
			for (int j = 0; j < nbCol; j++) {
				result[i + j * nbRow] = d[i][j];
			}
		}
		return Matrices.sparse(nbRow, nbCol, rowStart(nbRow, nbCol),
				columnStart(nbRow, nbCol), result);
	}

	/**
	 * Return the indexes inside a row of each element.
	 * 
	 * @param m
	 *            Number of rows
	 * @param n
	 *            Number of columns
	 * @return An array of indexes.
	 */
	private int[] rowStart(int m, int n) {
		int[] index = new int[m * n];
		for (int i = 0; i < m * n; i++) {
			index[i] = i % n;
		}
		return index;
	}

	/**
	 * Return the indexes of all the beginnings of a column for a matrix of size
	 * m*n.
	 * 
	 * @param n
	 *            Size of the matrix
	 * @return An array of indexes
	 */
	private int[] columnStart(int m, int n) {
		int[] index = new int[n + 1];
		for (int i = 0; i < n + 1; i++) {
			index[i] = i * m;
		}
		return index;
	}

	/**
	 * Transform a matrix to an array of arrays of doubles
	 * 
	 * @param w
	 *            The Matrix to transform
	 * @return An array of arrays of doubles
	 */
	private double[][] getValues2D(Matrix w) {
		int nbRow = w.numRows();
		int nbCol = w.numCols();
		double[][] sorted = new double[nbRow][nbCol];
		for (int i = 0; i < nbRow; i++) {
			for(int j=0; j< nbCol;j++){
				sorted[i][j] = w.apply(i,j);
			}
		}
		return sorted;
	}

}
