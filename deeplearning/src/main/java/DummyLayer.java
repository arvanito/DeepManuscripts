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

	public DummyLayer(ConfigBaseLayer configLayer) {
		super(configLayer, new DummyLearner(), new DummyExtractor(), new DummyPooler());
	}

}
