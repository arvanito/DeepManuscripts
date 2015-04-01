package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.mllib.linalg.Vector;

/**
 * A dummy Extractor - just passes the data through.
 * 
 * @author Arttu Voutilainen
 *
 */

public class DummyExtractor implements Extractor {

	private static final long serialVersionUID = -7897241422516586501L;
	
	@Override
	public Vector call(Vector data) throws Exception {
		return data;
	}

	@Override
	public void setConfigLayer(ConfigBaseLayer configLayer) {
	}
	
	@Override
	public void setFeatures(Vector[] features) {
	}

}
