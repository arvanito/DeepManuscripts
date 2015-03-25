package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

/**
 * 
 * A base implementation of a Layer. The constructor is given instances of the three classes,
 * Learner, Extractor and Pooler. For feature learning, we use the 'call'-function of the Learner class,
 * and for feature extraction and pooling we map the data through the respective classes.  
 * 
 * @author Arttu Voutilainen
 *
 */
public class BaseLayer implements DeepLearningLayer {

	ConfigBaseLayer configLayer;
	//PreProcessor preprocess;
	Learner learn;
	Extractor extract;
	Pooler pool;
	
	
	public BaseLayer(ConfigBaseLayer configLayer, Learner learn, Extractor extract, Pooler pool) {
		this.configLayer = configLayer;
		
		//this.preprocess = preprocess;
		this.learn = learn;
		this.extract = extract;
		this.pool = pool;
	}
	
	
	/*
	 @Override
	 public JavaRDD<Vector> preProcess(JavaRDD<Vector> data) {
	 	return preprocess.preprocessData(data);
	 }
	 */
	
	
	@Override
	public Vector[] learnFeatures(JavaRDD<Vector> data) throws Exception {
		return learn.call(data);
	}

	@Override
	public JavaRDD<Vector> extractFeatures(JavaRDD<Vector> data, ConfigBaseLayer configLayer, Vector[] features) {
		extract.setConfigLayer(configLayer);
		extract.setFeatures(features);
		return data.map(extract);
	}

	@Override
	public JavaRDD<Vector> pool(JavaRDD<Vector> data) {
		return data.map(pool);
	}

    @Override
	public JavaRDD<Vector> execute(JavaRDD<Vector> data) throws Exception {
		Vector[] features = learnFeatures(data);
		JavaRDD<Vector> represent = extractFeatures(data, configLayer, features);
		JavaRDD<Vector> pooled = pool(represent);
		return pooled;
	}

}
