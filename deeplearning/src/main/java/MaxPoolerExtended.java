package main.java;

import main.java.DeepModelSettings.ConfigFeatureExtractor;
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
public class MaxPoolerExtended implements Pooler {

	private static final long serialVersionUID = 1505267555549652215L;

	private int poolSize;

	protected int inputDim1;
	protected int inputDim2;
	
	protected boolean poolOver2DInput;
	
	
	/**
	 * Constructor that sets the current base layer configuration.
	 * 
	 * @param configLayer Current base layer configuration.
	 */
	public MaxPoolerExtended(ConfigBaseLayer configLayer) {
		
		poolSize = configLayer.getConfigPooler().getPoolSize();
		poolOver2DInput = false;
		
		// check if we pool over 2-d or 1-d
		if (configLayer.hasConfigFeatureExtractor()) {
			ConfigFeatureExtractor cf = configLayer.getConfigFeatureExtractor();
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
	 * It pools over every 'poolSize' contiguous elements
	 * of the vector by keeping only the one with the maximum value.
	 * 
	 * @param data 1D vector of doubles
	 * @return 1D vector with reduced size
	 */
	public Vector poolOver1D(Vector data)  {
		// The size of the new pooled vector
		int n = data.size()/poolSize;
		// Check that the pool size is not too big
		assert n >= 1: "Consider reducing the pool size";
		
		double[] pooled_data = new double[n];
		for (int i = 0; i < n; ++i) {
			// Find minimum in patch
			double maxPatch = -Double.MAX_VALUE;
			for (int k = 0; k < poolSize; ++k) {
			  maxPatch = Math.max(maxPatch, data.apply(i*poolSize + k));
			}
			pooled_data[i] = maxPatch;
		}
		return Vectors.dense(pooled_data);
	}
	/**
	 * The method shrinks the input array.
	 * It pools over every 'pool_size x pool_size' block
	 * of the corresponding matrix by keeping only the one with the maximum value.
	 * 
	 * @param data 1D vector storing column-wise a 2D image/feature patch
	 * @return 1D vector with reduced size storing the pooled data
	 */
	public Vector poolOver2D(Vector data)  {
		

		int output_dim1 = (int)Math.floor((double)inputDim1 / poolSize);
		int output_dim2 = (int)Math.floor((double)inputDim2 / poolSize);
		int nbr_features = data.size() / (inputDim1*inputDim2);
		double[] output = new double[output_dim1 * output_dim2* nbr_features];
		
		for (int f = 0; f < nbr_features; ++f) {
			int foffset = inputDim1*inputDim2* f;
			int fnewoffset = output_dim1*output_dim2*f;
			for (int i = 0; i < output_dim1; ++i) {
				for (int j = 0; j < output_dim2; ++j) {
					double maxPatch = -Double.MAX_VALUE;
					for (int ki = 0; ki < poolSize; ++ki) {
						for (int kj = 0; kj < poolSize; ++kj) {
							int lookup_i = i*poolSize + ki;
							int lookup_j = j*poolSize + kj;
							maxPatch = Math.max(maxPatch, data.apply(foffset + lookup_j*inputDim1 + lookup_i));
						}
					}
					output[fnewoffset + j*output_dim1 + i] = maxPatch;
				}
			}
		}
		return Vectors.dense(output);
	}

	public boolean isPoolOver2DInput() {
		return poolOver2DInput;
	}
	public int getInputDim1() {
		return inputDim1;
	}
	public int getInputDim2() {
		return inputDim2;
	}
}
