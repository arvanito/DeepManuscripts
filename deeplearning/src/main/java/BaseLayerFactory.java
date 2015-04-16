package main.java;


import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.BaseLayer;

public class BaseLayerFactory {

	static public BaseLayer createBaseLayer(ConfigBaseLayer configLayer) {
	//	super(configLayer, new PreProcessZCA(), new DummyLearner(), new DummyExtractor(), new MaxPooler(configLayer));
		PreProcessZCA preprocessor = null;
		if (configLayer.hasConfigPreprocess()) {
					preprocessor = new PreProcessZCA();
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
		}
		ConfigPooler cpooler = configLayer.getConfigPooler();
		Pooler pooler = null;
		if (cpooler.getPoolType() == ConfigPooler.PoolType.MAX) {
			pooler = new MaxPooler(configLayer);
		}
		System.out.printf("Pool size %d\n", cpooler.getPoolSize());

		return new BaseLayer(configLayer, preprocessor, learner, extractor, pooler);
	}
}
