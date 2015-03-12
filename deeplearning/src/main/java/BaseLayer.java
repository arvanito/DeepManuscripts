package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

/**
 * 
 * A base implementation of a Layer. The constructor is given instances of the three classes,
 * Learner, Extractor and Pooler. For feature extraction, we use the 'call'-function of the Pooler class,
 * and for feature extraction and pooling we map the data through the respective classes.  
 * 
 * @author Arttu Voutilainen
 *
 */
public class BaseLayer implements DeepLearningLayer {

	Learner learn;
	Extractor extract;
	Pooler pool;
	
	public BaseLayer(Learner learn, Extractor extract, Pooler pool) {
		this.learn = learn;
		this.extract = extract;
		this.pool = pool;
	}
	
	@Override
	public Vector[] learnFeatures(JavaRDD<Vector> data) throws Exception {
		return learn.call(data);
	}

	@Override
	public JavaRDD<Vector> extractFeatures(JavaRDD<Vector> data, Vector[] features) {
		return data.map(extract);
	}

	@Override
	public JavaRDD<Vector> pool(JavaRDD<Vector> data) {
		return data.map(pool);
	}

}
