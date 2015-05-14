package main.java;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;

public class KNearestNeighbor {
	private Vector[] input;
	private Matrix outputD;
	private Matrix outputS;
	private CoordinateMatrix outputC;
	private double maxDistance;
	private int matrixType;
	private int k;
	private int neighborType;
	private double sigma;

	/**
	 * @return the dense weighted Matrix
	 */
	public Matrix getDenseWeightedMatrix() {
		if (outputD == null) {
			this.compute();
		}
		return outputD;
	}

	/**
	 * @return the sparse weighted Matrix
	 */
	public Matrix getSparseWeightedMatrix() {
		if (outputS == null) {
			this.compute();
		}
		return outputS;
	}
	
	/**
	 * @return the weighted CoordinateMatrix
	 */
	public CoordinateMatrix getWeightedCoordinateMatrix() {
		if (outputC == null) {
			this.compute();
		}
		return outputC;
	}
	
	/**
	 * Constructor for WeightedMatrix
	 * 
	 * @param input
	 *            Array of Vectors
	 * @param k
	 *            Number of nearest neighbors
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
	public KNearestNeighbor(Vector[] input, int k, int matrixType,
			int neighborType, double sigma) {
		this.input = input;
		this.maxDistance = 0;
		this.matrixType = matrixType;
		this.k = k;
		this.neighborType = neighborType;
		this.sigma = sigma;
	}

	/**
	 * Smaller constructor for WeightedMatrix
	 * 
	 * @param input
	 *            Array of Vectors
	 * @param k
	 *            Number of nearest neighbors
	 * @param sigma
	 *            The value of sigma for the Gaussian similarity function
	 */
	public KNearestNeighbor(Vector[] input, int k, double sigma) {
		this.input = input;
		this.maxDistance = 0;
		this.matrixType = 1;
		this.k = k;
		this.neighborType = 1;
		this.sigma = sigma;
	}

	/**
	 * Smallest constructor for WeightedMatrix
	 * 
	 * @param input
	 *            Array of Vectors
	 * @param k
	 *            Number of nearest neighbors
	 */
	public KNearestNeighbor(Vector[] input, int k) {
		this.input = input;
		this.maxDistance = 0;
		this.matrixType = 1;
		this.k = k;
		this.neighborType = 1;
		this.sigma = 1;
	}

	/**
	 * Compute the weighted matrix from the input array of Vectors.
	 */
	public void compute() {
		double[][] distances = computeDistanceMatrix();
		double[][] kKeighbour = computeKNeighbours(distances);
		if (neighborType == 1) {
			kKeighbour = atMostK(kKeighbour);
		} else if (neighborType == 2) {
			kKeighbour = atLeastK(kKeighbour);
		}
		int n = input.length;
		double[] result = new double[n * n];
		if (matrixType == 0) {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					double w = 0;
					if (kKeighbour[i][j] > 0) {
						w = 1;
					}
					result[i + j * n] = w;
				}
			}
		} else {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					double w = gaussian(kKeighbour[i][j]);
					result[i + j * n] = w;
				}
			}
		}
		outputD = Matrices.dense(n, n, result);
		outputS = Matrices.sparse(n, n, columnStart(n), rowStart(n), result);
		outputC = coordinateMatrix(n,n,result);
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
	 * Return the indexes in the row of all the elements.
	 * 
	 * @param n
	 *            Size of the matrix
	 * @return An array of indexes
	 */
	private int[] rowStart(int n) {
		int[] index = new int[n * n];
		for (int i = 0; i < n * n; i++) {
			index[i] = i % n;
		}
		return index;
	}

	/**
	 * Return the indexes of all the beginnings of a column for a flatten square
	 * matrix of size n.
	 * 
	 * @param n
	 *            Size of the matrix
	 * @return An array of indexes
	 */
	private int[] columnStart(int n) {
		int[] index = new int[n + 1];
		for (int i = 0; i < n + 1; i++) {
			index[i] = i * n;
		}
		return index;
	}

	/**
	 * Compute the Gaussian similarity function for non-zero values.
	 * 
	 * @param d
	 *            Input value
	 * @return Calculated value
	 */
	private double gaussian(double d) {
		if (d == 0)
			return 0;
		return Math.exp(-Math.pow(d, 2) / (2 * Math.pow(sigma, 2)));
	}

	/**
	 * Compute a symmetric two-dimensional array from a non-symmetric
	 * two-dimensional array M by taking the maximum between M and
	 * M<sup>T</sup>.
	 * 
	 * @param M
	 *            The input non-symmetric two-dimensional array.
	 * @return A symmetric two-dimensional array
	 */
	private double[][] atLeastK(double[][] M) {
		double[][] result = M;
		int size = result.length;
		for (int i = 0; i < size; i++) {
			for (int j = i; j < size; j++) {
				double a = Math.max(result[i][j], result[j][i]);
				result[i][j] = a;
				result[j][i] = a;
			}
		}
		return result;
	}

	/**
	 * Compute a symmetric two-dimensional array from a non-symmetric
	 * two-dimensional array M by taking the minimum between M and
	 * M<sup>T</sup>.
	 * 
	 * @param M
	 *            The input non-symmetric two-dimensional array.
	 * @return A symmetric two-dimensional array
	 */
	private double[][] atMostK(double[][] M) {
		double[][] result = M;
		int size = result.length;
		for (int i = 0; i < size; i++) {
			for (int j = i; j < size; j++) {
				double a = Math.min(result[i][j], result[j][i]);
				result[i][j] = a;
				result[j][i] = a;
			}
		}
		return result;
	}

	/**
	 * Compute a non-symmetric matrix by keeping for each row the distance of
	 * the <i>k</i> nearest points, and setting the rest to 0.
	 * 
	 * @param distances
	 * @return
	 */
	private double[][] computeKNeighbours(double[][] distances) {
		int size = distances.length;
		double[][] results = distances;
		if (k >= size - 1) {
			return results;
		} else {
			for (int i = 0; i < size; i++) {
				double[][] toSort = new double[size][2];
				for (int j = 0; j < size; j++) {
					toSort[j][1] = j;
					toSort[j][0] = results[i][j];
				}
				double[][] sorted = sortWeights(toSort);
				for (int j = k + 1; j < size; j++) {
					results[i][(int) sorted[j][1]] = 0;
				}
			}
			return results;
		}
	}

	/**
	 * Function to sort a two-dimensionnal array
	 * 
	 * @param toSort
	 *            the array to sort
	 * @return The sorted array
	 */
	private double[][] sortWeights(double[][] toSort) {
		Arrays.sort(toSort, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				return Double.compare(o1[0], o2[0]);
			}
		});
		return toSort;
	}

	/**
	 * Compute the matrix of the distances between all the vectors of the
	 * <i>input</i> array.
	 * 
	 * @return A 2-dimensional array of double of the distances.
	 */
	private double[][] computeDistanceMatrix() {
		int n = input.length;
		double result[][] = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				double d = computeDistance(input[i], input[j]);
				result[i][j] = d;
				result[j][i] = d;
			}
		}
		return result;
	}

	/**
	 * Compute the distance between two vectors.
	 * 
	 * @param vector
	 * @param vector2
	 * @return The distance between the two vectors.
	 */
	private double computeDistance(Vector vector, Vector vector2) {
		int k = vector.size();
		int d = 0;
		double[] v1 = vector.toArray();
		double[] v2 = vector2.toArray();
		for (int i = 0; i < k; i++) {
			d += Math.pow(v1[i] - v2[i], 2);
		}
		double result = Math.sqrt(d);
		maxDistance = Math.max(maxDistance, result);
		return result;
	}
}
