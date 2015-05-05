package main.java;

import java.util.Arrays;
import java.util.Comparator;

import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;

public class WeightedMatrix {
	private Vector[] input;
	private Matrix output;
	private double maxDistance;
	private int type;
	private int k;

	/**
	 * @return the weighted Matrix
	 */
	public Matrix getWeightedMatrix() {
		if (output == null) {
			this.compute();
		}
		return output;
	}

	public WeightedMatrix(Vector[] input, int k, int type) {
		this.input = input;
		this.maxDistance = 0;
		this.type = type;
		this.k = k;
	}

	/**
	 * Compute the weighted matrix from the input array of Vectors.
	 */
	public void compute() {
		double[][] distances = computeDistanceMatrix();
		double[][] kKeighbour = computeKNeighbours(distances);
		int n = input.length;
		double[] result = new double[n * n];
		if (type == 0) {
			for (int i = 0; i < n; i++) {
				for (int j = i; j < n; j++) {
					double w = 0;
					if (distances[i][j] < maxDistance / 2) {
						w = 1;
					}
					result[i + j * n] = w;
					result[j + i * n] = w;
				}
			}
		} else {
			for (int i = 0; i < n; i++) {
				for (int j = i; j < n; j++) {
					double w = (maxDistance - distances[i][j]) / maxDistance;
					result[i + j * n] = w;
					result[j + i * n] = w;
				}
			}
		}
		output = Matrices.dense(n, n, result);
	}

	private double[][] computeKNeighbours(double[][] distances) {
		int size = distances.length;
		double[][] results = invertMatrix(distances);
		if (k >= size - 1) {
			return results;
		} else {
			for (int i = 0; i < size; i++) {
				double[][] toSort = new double[size][2];
				for (int j = 0; j < size; j++) {
					toSort[j][2] = j;
					toSort[j][1] = results[i][j];
				}
				double[][] sorted = sortWeights(toSort);
				for (int j = 0; j < size - k - 1; j++) {
					results[i][(int) sorted[j][2]] = 0;
				}
			}
			return results;
		}
	}

	private double[][] sortWeights(double[][] toSort) {
		Arrays.sort(toSort, new Comparator<double[]>() {
			public int compare(double[] o1, double[] o2) {
				return Double.compare(o1[0], o2[0]);
			}
		});
		return toSort;
	}

	private double[][] invertMatrix(double[][] distances) {
		int size = distances.length;
		double[][] results = new double[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = i; j < size; j++) {
				double w = (maxDistance - distances[i][j]) / maxDistance;
				results[i][j] = w;
				results[j][i] = w;
			}
		}
		return results;
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
	 * @return The distance bettwen the two vectors.
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
