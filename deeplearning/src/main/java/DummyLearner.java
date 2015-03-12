package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

/**
 * A dummy Learner - just passes the data through (turns it into a Vector array first).
 * 
 * @author Arttu Voutilainen
 *
 */

public class DummyLearner implements Learner {

	private static final long serialVersionUID = -8894452883883815463L;

	@Override
	public Vector[] call(JavaRDD<Vector> data) throws Exception {
		
		return data.collect().toArray(new Vector[0]);
	}

}
