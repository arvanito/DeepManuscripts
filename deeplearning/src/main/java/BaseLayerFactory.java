package main.java;


import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.BaseLayer;

/**
 * 
 * A Factory class for BaseLayers. Given the configuration for a base layer,
 * the class is responsible for properly instantiating  a derived object of type BaseLayer.  
 * 
 * @author Viviana Petrescu
 *
 */

public class BaseLayerFactory {

	/**
	 * Main method that creates a BaseLayer from the loaded protocol buffer configuration.
	 * 
	 * @param configLayer ConfigBaseLayer object read from the protocol buffer file
	 * @param layer_index Index that represents the number of layer created
	 * @param pathPrefix Path prefix for saving the trained model
	 * @return BaseLayer object that represents the current layer
	 */
	static public BaseLayer createBaseLayer(ConfigBaseLayer configLayer, int layer_index, String pathPrefix) {
		
		// set up the preprocessor
		PreProcessZCA preprocessor = null;
		if (configLayer.hasConfigPreprocess()) {
					preprocessor = new PreProcessZCA(configLayer);
		}
		
		// Assert one of this two needs to be true. We should have either K-means or Auto-encoders
		assert(configLayer.hasConfigAutoencoders() || configLayer.hasConfigKmeans());
		
		// set up the learner
		Learner learner = null;
		if (configLayer.hasConfigKmeans()) {
			learner = new KMeansLearner(configLayer);
			System.out.printf("Kmeans clusters %d\n", configLayer.getConfigKmeans().getNumberOfClusters());
		}
		if (configLayer.hasConfigAutoencoders()) {
			System.out.printf("Autoencoders units %d\n", configLayer.getConfigAutoencoders().getNumberOfUnits());
			// Not yet implemented
			System.out.printf("WARNING - WARNING - not yet implemented");
		}
		
		// set up the extractor, depending on the layer, either FFT convolutional extraction, 
		// or matrix-vector multiplication extraction
		Extractor extractor = null;
		if (configLayer.hasConfigFeatureExtractor()) {
				extractor = new FFTConvolutionExtractor(configLayer, preprocessor);
		} else {
				extractor = new MultiplyExtractor(configLayer, preprocessor);
		}
		
		// set up the pooler, by default max-pooler
		ConfigPooler cpooler = configLayer.getConfigPooler();
		Pooler pooler = null;
		if (cpooler.getPoolType() == ConfigPooler.PoolType.MAX) {
			pooler = new MaxPoolerExtended(configLayer);
		}
		System.out.printf("Pool size %d\n", cpooler.getPoolSize());
 
		// create and return the base layer
		BaseLayer b = new BaseLayer(configLayer, preprocessor, learner, extractor, pooler);
		b.setLayerIndex(layer_index);
		//TODO change this
		b.setPathPrefix(pathPrefix + Integer.toString(layer_index));
		return b;
	}
}
