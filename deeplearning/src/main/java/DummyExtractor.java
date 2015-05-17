package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
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
	public Tuple2<Vector, Vector> call(Tuple2<Vector, Vector> data) throws Exception {
		return data;
	}

	@Override
	public void setConfigLayer(ConfigBaseLayer configLayer) {
	}
	
	@Override
	public void setFeatures(Vector[] features) {
	}

	@Override
	public void setPreProcessZCA(DenseMatrix zca, DenseVector mean) {
	}

	@Override
	public void setEps1(double eps1) {
		// TODO Auto-generated method stub
		
	}

}
