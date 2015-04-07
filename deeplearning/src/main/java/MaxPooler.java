package main.java;

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
	public MaxPooler(ConfigBaseLayer c) {
		if (c.hasConfigPooler()) {
			config = c.getConfigPooler();
		}
	}
	@Override
	public Vector call(Vector data) throws Exception {
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
	
	public ConfigPooler getConfig() {
		return config;
	}
	public void setConfig(ConfigPooler c) {
		config = c;
	}
}
