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
	protected int input_width;
	protected int input_height;
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
			//	cf.getFeatureDim1() < cf.getInputDim2() ||		
			//    cf.getFeatureDim2() < cf.getInputDim1() ||
			    cf.getFeatureDim2() < cf.getInputDim2()) {
				poolOver2DInput = true;
				input_width = cf.getInputDim1() - cf.getFeatureDim1() + 1;
				input_height = cf.getInputDim2() - cf.getFeatureDim2() + 1;
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
		//TODO(viviana) The pooler needs to know if the input vector comes from
		//  a 2D or 1D data  
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
		//TODO 
		return data;
	}
	public ConfigPooler getConfig() {
		return config;
	}
	public void setConfig(ConfigPooler c) {
		config = c;
	}
}
