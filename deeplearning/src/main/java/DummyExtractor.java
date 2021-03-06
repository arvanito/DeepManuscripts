package main.java;

import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
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
