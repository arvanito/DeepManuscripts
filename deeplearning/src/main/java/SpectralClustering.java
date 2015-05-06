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

public class SpectralClustering {

	Vector[] input;
	int k;
	KMeansModel training;
	JavaRDD<Integer> vectorClassification;

	/**
	 * General constructor for Spectral Clustering.
	 * 
	 * @param input
	 *            : Input array of Vectors to sort by clusters.
	 *            </ul>
	 * @param k
	 *            : Number of clusters to construct.
	 */
	public SpectralClustering(Vector[] input, int spectralType, int k) {
		this.input = input;
		this.k = k;
		this.vectorClassification = this.compute();
	}

	/**
	 * Train the Spectral Clustering Algorithm using the data provided by the
	 * Input Matrix.
	 * @return 
	 * 
	 */
	private JavaRDD<Integer> compute() {
		// TODO Modify to make more customizable.
		Matrix w = new KNearestNeighbor(input, 3,0,1,1).getWeightedMatrix();
		Matrix d = getDegreeMatrix(w);
		Matrix l = computeUnnormalizedLaplacian(w, d);
		// Computing the eigenvectors
		Matrix u = eigenvectorComputing(l);

		return kmeansTraining(u);
	}

	private JavaRDD<Integer> kmeansTraining(Matrix y) {
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

		SparkConf conf = new SparkConf().setAppName("K-means");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<Vector> distData = sc.parallelize(data);

		int numIterations = 20;
		KMeansModel clusters = KMeans.train(JavaRDD.toRDD(distData), k,
				numIterations);
		sc.close();
		this.training =  clusters;
		return clusters.predict(distData);
	}

	/**
	 * Normalize the rows of u to norm 1.
	 * 
	 * @param u
	 *            : The matrix containing the eigenvectors.
	 * @return A normalized matrix of eigenvectors.
	 */
	private Matrix normalize(Matrix u) {
		// TODO
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
		// TODO
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
		// TODO
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
