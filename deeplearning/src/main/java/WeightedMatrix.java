package main.java;

import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;

public class WeightedMatrix {
	private Vector[] input;
	private Matrix output;
	private double maxDistance;
	/**
	 * @return the weighted Matrix
	 */
	public Matrix getWeightedMatrix() {
		if(output == null){
			this.compute();
		}
		return output;
	}
	
	public WeightedMatrix(Vector[] input){
		this.input = input;
		this.maxDistance = 0;
	}
	
	/**
	 *  Compute the weighted matrix from the input array of Vectors.
	 */
	public void compute(){
		double[][] distances = computeDistanceMatrix();
		int n = input.length;
		double[] result = new double[n * n];
		for(int i = 0; i< n; i++){
			for(int j = i; j<n;j++){
				double w = (maxDistance - distances[i][j])/maxDistance;
				result[i + j * n] = w;
				result[j + i * n] = w;
			}
		}
		output = Matrices.dense(n, n, result);
	}
	
	/**
	 * Compute the matrix of the distances between all the vectors of the <i>input</i> array.
	 * @return A 2-dimensional array of double of the distances.
	 */
	private double[][] computeDistanceMatrix(){
		int n = input.length;
		double result[][] = new double[n][n];
		for(int i= 0; i<n;i++){
			for(int j= i;j<n;j++){
				double d = computeDistance(input[i],input[j]);
				result[i][j] = d;
				result[j][i] = d;
			}
		}
		return result;
	}

	/**
	 *  Compute the distance between two vectors.
	 * @param vector
	 * @param vector2
	 * @return The distance bettwen the two vectors.
	 */
	private double computeDistance(Vector vector, Vector vector2) {
		int k = vector.size();
		int d = 0;
		double[] v1 = vector.toArray();
		double[] v2 = vector2.toArray();
		for(int i=0; i<k; i++){
			d += Math.pow(v1[i]-v2[i], 2);
		}
		double result = Math.sqrt(d);
		maxDistance = Math.max(maxDistance, result);
		return result;
	}
}
