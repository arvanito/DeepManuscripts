package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

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
	
	PreProcessor preprocess;
	Learner learn;
	Extractor extract;
	Pooler pool;
	
	// The path to the folder of the .prototxt file, where the 
	// weight will be saved. (This path should already exist.)
	String pathPrefix;
	
	// The layer number, starting from 0
	int layer_index;
	// Ugly hack. Spark context is needed by the preprocessor.
	// It needs to parallelize a DenseMatrix object.
	JavaSparkContext spark_context; 
	
	// By default, saving of the model is false
	boolean save_model;
	
	
	public BaseLayer(ConfigBaseLayer configLayer, PreProcessor preprocess, Learner learn, Extractor extract, Pooler pool) {
		this.configLayer = configLayer;
		
		this.preprocess = preprocess;
		this.learn = learn;
		this.extract = extract;
		this.pool = pool;
		
		save_model = false;
	}
	
	
	@Override
	public JavaRDD<Tuple2<Vector, Vector>> preProcess(JavaRDD<Tuple2<Vector, Vector>> data) {
		return preprocess.preprocessData(data);
	}
	
	
	@Override
	public Vector[] learnFeatures(JavaRDD<Tuple2<Vector, Vector>> data) throws Exception {
		return learn.call(data);
	}

	
	@Override
	public JavaRDD<Tuple2<Vector, Vector>> extractFeatures(JavaRDD<Tuple2<Vector, Vector>> data, 
										ConfigBaseLayer configLayer, Vector[] features) {
		extract.setConfigLayer(configLayer);
		extract.setFeatures(features);
		return data.map(extract);
	}

	
	@Override
	public JavaRDD<Tuple2<Vector, Vector>> pool(JavaRDD<Tuple2<Vector, Vector>> data) {
		return data.map(pool);
	}

	
	//TODO:: Input two datasets, one is patch-based and the other is larger parts of the image.
    @Override
	public JavaRDD<Tuple2<Vector, Vector>> train(JavaRDD<Tuple2<Vector, Vector>> input_small_patches, 
			                       JavaRDD<Tuple2<Vector, Vector>> input_word_patches) throws Exception {
    	
    	JavaRDD<Tuple2<Vector, Vector>> preprocessed = preProcess(input_small_patches);
    	if (save_model == true)
    		this.preprocess.saveToFile(pathPrefix + "_preprocess", spark_context);
    	
		Vector[] features = learnFeatures(preprocessed);
		
		// TODO:: do preprocessing on the second dataset
		//JavaRDD<Vector> preprocessedBig = dataBig.map(preprocess);
		
		// Ugly hack, move this to the Learner class
		if (save_model == true)
			LinAlgebraIOUtils.saveVectorArrayToObject(features, pathPrefix + "_features", spark_context);
		
		JavaRDD<Tuple2<Vector, Vector>> represent = extractFeatures(input_word_patches, configLayer, features);
		JavaRDD<Tuple2<Vector, Vector>> pooled = pool(represent);
		return pooled;
	}

    
    public JavaRDD<Tuple2<Vector, Vector>> test(JavaRDD<Tuple2<Vector, Vector>> data) throws Exception {
    	// Setup the preprocessor
    	this.preprocess.loadFromFile(pathPrefix + "_preprocess", spark_context);
//    	JavaRDD<Vector> preprocessed = preProcess(data);
    	
    	//TODO load the features from file.
    	Vector[] features = LinAlgebraIOUtils.loadVectorArrayFromObject(pathPrefix + "_features", spark_context);

    	System.out.println("Features info");
    	System.out.println(features.length);
    	System.out.println(features[0].size());
    	
    	JavaRDD<Tuple2<Vector, Vector>>represent = extractFeatures(data, configLayer, features);
    	JavaRDD<Tuple2<Vector, Vector>> pooled = pool(represent);
    	return pooled;
    }
   
    
    public String getPathPrefix() {
    	return pathPrefix;
    }
    public void setPathPrefix(String s) {
    	pathPrefix = s;
    }
    public int getLayerIndex() {
    	return layer_index;
    }
    public void setLayerIndex(int l) {
    	layer_index = l;
    }
    public void setSparkContext(JavaSparkContext sc) {
    	spark_context = sc;
    }
	public void setSaveModel(boolean value) {
		save_model = value;
	}
	public boolean getSaveModel() {
		return this.save_model;
	}
}
