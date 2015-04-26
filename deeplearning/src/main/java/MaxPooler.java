package main.java;

import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.DeepModelSettings.ConfigBaseLayer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * A max Pooler 
 * 
 * @author Viviana Petrescu
 *
 */
public class MaxPooler implements Pooler {

	private static final long serialVersionUID = 1505267555549652215L;
	protected ConfigPooler config;
	// Only relevant for 2D input
	protected int input_dim1;
	protected int input_dim2;
	/**
	 *  The field is true if the input Vector to the method call
	 *  comes from a 2D matrix
	 */
	protected boolean poolOver2DInput;
	public MaxPooler(ConfigBaseLayer c) {
		if (c.hasConfigPooler()) {
			config = c.getConfigPooler();
		}
		poolOver2DInput = false;
		// Look at the extractor and see if it is pooling over 2d or 1d input
		if (c.hasConfigFeatureExtractor()) {
			ConfigFeatureExtractor cf = c.getConfigFeatureExtractor();
			if (cf.getFeatureDim1() < cf.getInputDim1() ||
			    cf.getFeatureDim2() < cf.getInputDim2()) {
				poolOver2DInput = true;
				input_dim1 = cf.getInputDim1() - cf.getFeatureDim1() + 1;
				input_dim2 = cf.getInputDim2() - cf.getFeatureDim2() + 1;
			}
		}
	}
	@Override
	public Vector call(Vector data) throws Exception {
		//TODO(viviana) The pooler needs to know if the input vector comes from
		//  a 2D or 1D data  
		if (this.poolOver2DInput == false)
			return poolOver1D(data);
		else
			return poolOver2D(data);
	}
	/**
	 * The method shrinks the input array.
	 * It pools over every 'pool_size' contiguous elements
	 * of the vector by keeping only the one with the maximum value.
	 * 
	 * @param data 1D vector of doubles
	 * @return 1D vector with reduced size
	 */
	public Vector poolOver1D(Vector data)  {
		int pool_size = config.getPoolSize();
		// The size of the new pooled vector
		int n = data.size()/pool_size;
		// Check that the pool size is not too big
		assert n >= 1: "Consider reducing the pool size";
		
		double[] pooled_data = new double[n];
		for (int i = 0; i < n; ++i) {
			// Find minimum in patch
			double maxPatch = -Double.MAX_VALUE;
			for (int k = 0; k < pool_size; ++k) {
			  maxPatch = Math.max(maxPatch, data.apply(i*pool_size + k));
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
		
		int pool_size = config.getPoolSize();

		int output_dim1 = (int)Math.floor((double)input_dim1 / pool_size);
		int output_dim2 = (int)Math.floor((double)input_dim2 / pool_size);
		double[] output = new double[output_dim1 * output_dim2];
		
		for (int i = 0; i < output_dim1; ++i) {
			for (int j = 0; j < output_dim2; ++j) {
				double maxPatch = -Double.MAX_VALUE;
				for (int ki = 0; ki < pool_size; ++ki) {
					for (int kj = 0; kj < pool_size; ++kj) {
					  int lookup_i = i*pool_size + ki;
					  int lookup_j = j*pool_size + kj;
					  maxPatch = Math.max(maxPatch, data.apply(lookup_j*input_dim1 + lookup_i));
					}
				}
				output[j*output_dim1 + i] = maxPatch;
			}
		}
		return Vectors.dense(output);
	}
	public ConfigPooler getConfig() {
		return config;
	}
	public void setConfig(ConfigPooler c) {
		config = c;
	}
	public boolean isPoolOver2DInput() {
		return poolOver2DInput;
	}
	public int getInputDim1() {
		return input_dim1;
	}
	public int getInputDim2() {
		return input_dim2;
	}
}
