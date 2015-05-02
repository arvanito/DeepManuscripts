package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 * A dummy Extractor - just passes the data through.
 * 
 * @author Arttu Voutilainen
 *
 */

public class DummyExtractor implements Extractor {

	private static final long serialVersionUID = -7897241422516586501L;
	
	@Override
	public Tuple2<Vector,Vector> call(Tuple2<Vector,Vector> data) throws Exception {
		return data;
	}

	@Override
	public void setConfigLayer(ConfigBaseLayer configLayer) {
	}
	
	@Override
	public void setFeatures(Vector[] features) {
	}

}
