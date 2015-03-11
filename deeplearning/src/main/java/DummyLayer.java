package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

/**
 * 
 * A dummy implementation of a Layer. Just passes the data through.
 *
 */
public class DummyLayer implements DeepLearningLayer {

	@Override
	public JavaRDD<Vector> learnFeatures(JavaRDD<Vector> data) {
		return data;
	}

	@Override
	public JavaRDD<Vector> extractFeatures(JavaRDD<Vector> data,
			JavaRDD<Vector> features) {
		return data;
	}

	@Override
	public JavaRDD<Vector> pool(JavaRDD<Vector> data) {
		return data;
	}

}
