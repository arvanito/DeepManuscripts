package main.java;

import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

/**
 * A max Pooler 
 * 
 * @author Viviana Petrescu
 *
 */
public class MaxPooler implements Pooler {

	private static final long serialVersionUID = 1505267555549652215L;
	
	private int poolSize;

	protected int inputDim1;
	protected int inputDim2;

	protected boolean poolOver2DInput;

	
	/**
	 * Constructor that sets the current base layer configuration.
	 * 
	 * @param c Current base layer configuration.
	 */
	public MaxPooler(ConfigBaseLayer c) {
		
		poolSize = c.getConfigPooler().getPoolSize();
		poolOver2DInput = false;
		
		// check if we pull over 1-d or 2-d
		if (c.hasConfigFeatureExtractor()) {
			ConfigFeatureExtractor cf = c.getConfigFeatureExtractor();
			if (cf.getFeatureDim1() < cf.getInputDim1() || cf.getFeatureDim2() < cf.getInputDim2()) {
				poolOver2DInput = true;
				
				// convolutional dimensions
				inputDim1 = cf.getInputDim1() - cf.getFeatureDim1() + 1;
				inputDim2 = cf.getInputDim2() - cf.getFeatureDim2() + 1;
			}
		}
	}
	
	
	/**
	 * Main method that calls 1-d or 2-d pooling over the data point.
	 * 
	 * @param pair Input tuple data.
	 * @return Pooled data.
	 */
	@Override
	public Tuple2<Vector, Vector> call(Tuple2<Vector, Vector> pair) {

		// pool over 1-d or 2-d input
		Vector pooledVec = null;
		if (this.poolOver2DInput == false)
			pooledVec = poolOver1D(pair._2);
		else
			pooledVec = poolOver2D(pair._2);
		
		return new Tuple2<Vector, Vector>(pair._1, pooledVec);
	}
	
	
	/**
	 * The method shrinks the input array.
	 * It pools over every 'pool_size' contiguous elements
	 * of the vector by keeping only the one with the maximum value.
	 * 
	 * @param data Input 1-d data.
	 * @return Pooled 1-d vector.
	 */
	public Vector poolOver1D(Vector data)  {
		
		// the size of the new pooled vector
		int n = data.size() / poolSize;
		
		// Check that the pool size is not too big
		assert n >= 1: "Consider reducing the pool size";
		
		// perform pooling
		double[] pooled_data = new double[n];
		for (int i = 0; i < n; i++) {
			
			// find minimum in patch
			double maxPatch = -Double.MAX_VALUE;
			for (int k = 0; k < poolSize; k++) {
			  maxPatch = Math.max(maxPatch, data.apply(i * poolSize + k));
			}
			pooled_data[i] = maxPatch;
		}
		
		return Vectors.dense(pooled_data);
	}
	
	
	/**
	 * The method shrinks the input array. 
	 * It pools over every 'poolSize x poolSize' block
	 * of the corresponding matrix by keeping only the one with the maximum value.
	 * 
	 * @param data 1-d vector storing column-wise a series of 2-d patches.
	 * @return Pooled 1-d vector.
	 */
	public Vector poolOver2D(Vector data)  {
		
		int outputDim1 = (int) Math.floor((double)inputDim1 / poolSize);
		int outputDim2 = (int) Math.floor((double)inputDim2 / poolSize);
		
		// perform 2-d pooling
		double[] output = new double[outputDim1*outputDim2];
		for (int i = 0; i < outputDim1; i++) {
			for (int j = 0; j < outputDim2; j++) {
				
				// find the maximum over the current 2-d block
				double maxPatch = -Double.MAX_VALUE;
				for (int ki = 0; ki < poolSize; ki++) {
					for (int kj = 0; kj < poolSize; kj++) {
						int lookup_i = i * poolSize + ki;
						int lookup_j = j * poolSize + kj;
						maxPatch = Math.max(maxPatch, data.apply(lookup_j * inputDim1 + lookup_i));
					}
				}
				output[j * outputDim1 + i] = maxPatch;
			}
		}
		
		return Vectors.dense(output);
	}

	/**
	 * Method that returns the boolean indicating pooling dimensions.
	 * 	
	 * @return Boolean value.
	 */
	public boolean isPoolOver2DInput() {
		return poolOver2DInput;
	}
	
	
	/**
	 * Method that returns the first input dimension.
	 * 
	 * @return First input dimension.
	 */
	public int getInputDim1() {
		return inputDim1;
	}
	
	
	/**
	 * Method that returns the second input dimension.
	 * 
	 * @return Second input dimension.
	 */
	public int getInputDim2() {
		return inputDim2;
	}
	
}
