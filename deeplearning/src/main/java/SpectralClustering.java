package main.java;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealVector;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.rdd.RDD;

public class SpectralClustering {

	Matrix s;
	int spectralType;
	int k;
	int similarityGraph;
	int similarityGraphArgument;

	/**
	 * General constructor for Spectral Clustering
	 * 
	 * @param input
	 *            : Input similarity matrix. Must be square.
	 * @param spectralType
	 *            : Type of Spectral Clustering Algorithm that will be used:
	 *            <ul>
	 *            <li><i>0:</i> Uses Unnormalized spectral clustering.
	 *            <li><i>1:</i> Uses Normalized spectral clustering according to
	 *            Shi and Malik (2000)
	 *            <li><i>2:</i> Uses Normalized spectral clustering according to
	 *            Ng, Jordan, and Weiss (2002)
	 *            </ul>
	 * @param k
	 *            : Number of clusters to construct.
	 * @param similarityGraph
	 *            : Type of Similarity graphs that will be used:
	 *            <ul>
	 *            <li><i>0:</i> Uses <i>&epsilon;</i>-neighborhood graph .
	 *            <li><i>1:</i> Uses <i>k</i>-nearest neighbor graphs
	 *            <li><i>2:</i> Uses the fully connected graph
	 *            </ul>
	 * @param sGargument
	 *            : Argument for the similarity graphs.
	 */
	public SpectralClustering(Matrix input, int spectralType, int k,
			int similarityGraph, int sGargument) {
		this.s = input;
		this.spectralType = spectralType;
		this.k = k;
		this.similarityGraph = similarityGraph;
		this.similarityGraphArgument = sGargument;
	}

	/**
	 * Default constructor for Spectral Clustering using <i>k</i>-nearest
	 * neighbor graphs and unnormalized spectral clustering.
	 * 
	 * @param input
	 *            : Input similarity matrix. Must be square.
	 * @param k
	 *            : Number of clusters to construct.
	 * @param sGargument
	 *            : Argument for the similarity graphs.
	 */
	public SpectralClustering(Matrix input, int k, int sGargument) {
		this.s = input;
		this.spectralType = 0;
		this.k = k;
		this.similarityGraph = 1;
		this.similarityGraphArgument = sGargument;
	}

	/**
	 * Compute the Spectral Clustering Algorithm
	 * 
	 * @return Clusters A<sub>1</sub>,...,A<sub>k</sub> .
	 */
	public KMeansModel compute() {
		// Constructing similarity graphs
		Matrix w = null;
		switch (similarityGraph) {
		case 0:
			w = computeEpsilon();
			// TODO: Find an implementation or some papers
			break;
		case 1:
			w = computeKNearest();
			break;
		case 2:
			w = computeFullyConnected();
			// TODO: Find an implementation or some papers
			break;
		default:
			w = computeKNearest();
			break;
		}
		Matrix d = getDegreeMatrix(w);
		Matrix l = null;
		// Computing different Laplacian
		if (spectralType == 0 | spectralType == 1) {
			l = computeUnnormalizedLaplacian(w, d);
		}
		if (spectralType == 2) {
			l = computeLsym(w, d);
		}
		// Computing the eigenvectors
		Matrix u = null;
		if (spectralType == 0 | spectralType == 2) {
			u = eigenvectorComputing(l);
			if (spectralType == 2) {
				Matrix t = null;
				t = normalize(u);
				u = t;
			}
		}
		if (spectralType == 1) {
			u = generalizedEigenvectorComputing(l);
		}
		Matrix y = rows(u);

		return kmeans(y);
	}

	private KMeansModel kmeans(Matrix y) {
		int nbRow = y.numRows();
		int nbCol = y.numCols();
		double[][] yArray = getValues2D(y);
		Vector[] vectors = new Vector[nbRow];
		
		for(int i= 0; i<nbRow; i++){
			double[] temp = new double[nbCol];
			for(int j=0; j<nbCol; j++){
				temp[j] = yArray[i][j];
			}
			vectors[i] = Vectors.dense(temp);
		}
		List<Vector> data = Arrays.asList(vectors);
		
		SparkConf conf = new SparkConf().setAppName("K-means");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<Vector> distData = sc.parallelize(data);
		
		int numIterations = 20;
		KMeansModel clusters = KMeans.train(JavaRDD.toRDD(distData), k, numIterations);

		// Evaluate clustering by computing Within Set Sum of Squared Errors
		// double WSSSE = clusters.computeCost(parsedData.rdd());
		// System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
		return clusters;
	}

	/**
	 * Get the values <i>y<sub>i</sub></i> &isin; &real;<sup><i>k</i></sup> from
	 * the <i>i</i>-th row of u.
	 * 
	 * @param u
	 *            : The matrix containing the eigenvectors.
	 * @return Matrix of the <i>y<sub>i</sub></i> values.
	 */
	private Matrix rows(Matrix u) {
		return null;
	}

	/**
	 * Normalize the rows of u to norm 1.
	 * 
	 * @param u
	 *            : The matrix containing the eigenvectors.
	 * @return A normalized matrix of eigenvectors.
	 */
	private Matrix normalize(Matrix u) {
		return null;
	}

	/**
	 * Compute the k generalized eigenvectors of the matrix generalized
	 * eigenproblem lu=&lambda;du.
	 * 
	 * @param l
	 *            : Laplacian matrix.
	 * @return The k generalized eigenvectors.
	 */
	private Matrix generalizedEigenvectorComputing(Matrix l) {
		return null;
	}

	/**
	 * Compute the k eigenvectors of the matrix l.
	 * 
	 * @param l
	 *            : Laplacian matrix.
	 * @return The k eigenvectors of the matrix l.
	 */
	private Matrix eigenvectorComputing(Matrix l) {
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
	 * Compute the normalized symetric Laplacian
	 * 
	 * @param w
	 *            : Weighted adjacency matrix W.
	 * @param d
	 *            : Degree matrix D.
	 * @return Normalized graph Laplacian symetric matrix L<sub>sym</sub>
	 */
	private Matrix computeLsym(Matrix w, Matrix d) {
		return null;
	}

	/**
	 * Compute the unnormalized Laplacian
	 * 
	 * @param w
	 *            : Weighted adjacency matrix W.
	 * @param d
	 *            : Degree matrix D.
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
	 * Compute the Similarity matrix using the fully connected graph algorithm.
	 * 
	 * @return Similarity matrix
	 */
	private Matrix computeFullyConnected() {
		return null;
	}

	/**
	 * Compute the Similarity matrix using the <i>k</i>-nearest neighbor graphs
	 * algorithm.
	 * 
	 * @return Similarity matrix
	 */
	private Matrix computeKNearest() {
		// https://issues.apache.org/jira/browse/SPARK-2335
		return null;
	}

	/**
	 * Compute the Similarity matrix using the <i>&epsilon;</i>-neighborhood
	 * graph algorithm.
	 * 
	 * @return Similarity matrix
	 */
	private Matrix computeEpsilon() {
		return null;
	}

	/**
	 * Calculate the degree matrix D from the input matrix W.
	 * 
	 * @param w
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
	 *            : Number of rows
	 * @param nbCol
	 *            : Number of columns
	 * @param d
	 *            : Array of arrays of doubles
	 * @return The Matrix
	 */
	private Matrix toMatrix(int nbRow, int nbCol, double[][] d) {
		double[] result = new double[nbRow * nbCol];
		for (int i = 0; i < nbRow; i++) {
			for (int j = 0; j < nbCol; j++) {
				result[i + j * nbRow] = d[i][j];
			}
		}
		return Matrices.dense(nbRow, nbCol, result);
	}

	/**
	 * Transform a matrix to an array of arrays of doubles
	 * 
	 * @param w
	 *            : The Matrix to transform
	 * @return An array of arrays of doubles
	 */
	private double[][] getValues2D(Matrix w) {
		int nbRow = w.numRows();
		int nbCol = w.numCols();
		double[] stored = w.toArray();
		double[][] sorted = new double[nbRow][nbCol];
		for (int i = 0; i < stored.length; i++) {
			sorted[i % nbRow][(int) (i / nbCol)] = stored[i];
		}
		return sorted;
	}

}
