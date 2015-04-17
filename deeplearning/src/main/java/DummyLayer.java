package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

/**
 * 
 * A dummy implementation of a Layer. Just passes the data through, using Dummy(Learner,Extractor,Pooler).
 *
 * @author Arttu Voutilainen
 * 
 */
public class DummyLayer extends BaseLayer {
	/**
	 * @param configLayer holds the configuration for every type of layer
	 *  For accessing DummyLearner config:
	 * 		     configLayer.getConfigKmeans();
	 *			 configLayer.getConfigAutoencoders()
	 *  For accessing DummyPooler config
     *			 configLayer.getConfigPooler();
	 */
	public DummyLayer(ConfigBaseLayer configLayer) {
		super(configLayer, new PreProcessZCA(), new DummyLearner(), new DummyExtractor(), new MaxPooler(configLayer));
	}

}
