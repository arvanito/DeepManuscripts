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

	static public BaseLayer createBaseLayer(ConfigBaseLayer configLayer, int layer_index, String pathPrefix) {
		PreProcessZCA preprocessor = null;
		if (configLayer.hasConfigPreprocess()) {
					preprocessor = new PreProcessZCA(configLayer);
		}
		// Assert one of this two needs to be true.
		assert(configLayer.hasConfigAutoencoders()|| configLayer.hasConfigKmeans());
		
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
		Extractor extractor = null;
		if (configLayer.hasConfigFeatureExtractor()) {
				extractor = new FFTConvolutionExtractor(configLayer, preprocessor);
		} else {
				extractor = new MultiplyExtractor(configLayer, preprocessor);
		}
		ConfigPooler cpooler = configLayer.getConfigPooler();
		Pooler pooler = null;
		if (cpooler.getPoolType() == ConfigPooler.PoolType.MAX) {
			pooler = new MaxPoolerExtended(configLayer);
		}
		System.out.printf("Pool size %d\n", cpooler.getPoolSize());
 
		BaseLayer b = new BaseLayer(configLayer, preprocessor, learner, extractor, pooler);
		b.setLayerIndex(layer_index);
		//TODO change this
		b.setPathPrefix(pathPrefix + Integer.toString(layer_index));
		return b;
	}
}
