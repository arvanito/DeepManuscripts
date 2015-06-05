package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.storage.StorageLevel;

import scala.Tuple2;

/**
 * 
 * A base implementation of a Layer. The constructor is given instances of the four classes,
 * PreProcessor, Learner, Extractor and Pooler.
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
	
	// path prefix for saving the trained model
	String pathPrefix;
	
	// the layer number, starting from 0
	int layerIndex;
	
	// spark context is needed by the preprocessor.
	// ιt needs to parallelize a DenseMatrix object.
	JavaSparkContext sparkContext; 
	
	// indicates if we save the trained model, or not
	boolean saveModel;
	
	// indicates if we are the last layer, or not
	boolean notLast;
	
	
	/**
	 * Constructor of a BaseLayer. Sets the necessary class objects. TODO:: Change this to put more parameters inside!!
	 * 
	 * @param configLayer Base layer configuration.
	 * @param preprocess PreProcessor object.
	 * @param learn Learner object.
	 * @param extract Extractor object.
	 * @param pool Pooler object.
	 * @param layerIndex Current layer index.
	 */
	public BaseLayer(ConfigBaseLayer configLayer, PreProcessor preprocess, Learner learn, Extractor extract, Pooler pool, int layerIndex) {
		this.configLayer = configLayer;
		
		this.preprocess = preprocess;
		this.learn = learn;
		this.extract = extract;
		this.pool = pool;
		
		this.layerIndex = layerIndex;
	
		// by default, we do not save the model
		saveModel = false;
	}
	
	
	/**
	 * Overriden method for pre-processing the data. Calls the preprocessData from the PreProcessor class.
	 * 
	 * @param data Input distributed dataset.
	 * @return Preprocessed distributed dataset.
	 */
	@Override
	public JavaRDD<Tuple2<Vector, Vector>> preProcess(JavaRDD<Tuple2<Vector, Vector>> data) {
		return preprocess.preprocessData(data);
	}
	
	
	/**
	 * Overriden method for feature learning. Calls the call from the Learner class.
	 * 
	 * @param data Input distributed dataset.
	 * @return Array of Vector objects that represent the learned features.
	 */
	@Override
	public Vector[] learnFeatures(JavaRDD<Tuple2<Vector, Vector>> data) {
		return learn.call(data);
	}

	
	/**
	 * Overriden method for feature extraction. Does a map for every data point in the distributed dataset.
	 * 
	 * @param data Input distributed dataset.
	 * @param features Array of learned features from the learning step.
	 * @return New distributed dataset that contains the new representations of the data points.
	 */
	@Override
	public JavaRDD<Tuple2<Vector, Vector>> extractFeatures(JavaRDD<Tuple2<Vector, Vector>> data, Vector[] features) {
		
		// if we do pre-processing in the current layer, 
		// set the mean and ZCA variables to the Extractor
		// and the epsilon variable for the contrast normalization
		if(preprocess != null) {
			extract.setPreProcessZCA(((PreProcessZCA) preprocess).getMean(), ((PreProcessZCA) preprocess).getZCA());
			extract.setEps1(configLayer.getConfigPreprocess().getEps1());
		}
		
		// set the learned features
		extract.setFeatures(features);
		
		// do the map
		return data.map(extract);
	}

	
	/**
	 * Overriden method for pooling of representations. Does a map for every data point in the distributed dataset.
	 * 
	 * @param data Input distributed dataset.
	 * @return New pooled distributed dataset.
	 */
	@Override
	public JavaRDD<Tuple2<Vector, Vector>> pool(JavaRDD<Tuple2<Vector, Vector>> data) {
		return data.map(pool);
	}

	
	/**
	 * Overriden method that performs a complete training of one layer.
	 * 
	 * @param input1 Initial distributed dataset. In the first layer, it will contain small patches. 
	 * For the next layers, it will contain the pooled representations from the previous layer. 
	 * @param input2 Second distributed dataset. In the first layer, it ill contain large patches.
	 * For the next layers, it will be the same as the first input dataset.
	 * @return Distributed dataset containing the final pooled representations for the current layer.
	 * @throws Exception Standard Exception object.
	 */
	@Override
	public JavaRDD<Tuple2<Vector, Vector>> train(JavaRDD<Tuple2<Vector, Vector>> input1, JavaRDD<Tuple2<Vector, Vector>> input2) throws Exception {
    	
		// TODO:: make this more automatic!
    	int numPartitions = 400 * 4; 	// num-workers * cores_per_worker * succesive tasks
    	
		JavaRDD<Tuple2<Vector, Vector>> input1Preprocessed;
		Vector[] features;
		
		// pre-process data
		if (preprocess != null) {
			input1Preprocessed = preProcess(input1);
			
			// save the mean and ZCA variables
			if (saveModel == true) {
				preprocess.saveToFile(pathPrefix + "_preprocess_layer_" + layerIndex +"_" + System.currentTimeMillis(), sparkContext);
			}
		} else {
			input1Preprocessed = input1;
		}
    	
		// cache the pre-processed data, if possible
		StorageLevel storageLevel = input1Preprocessed.getStorageLevel();
		if (storageLevel == StorageLevel.NONE()){
			input1Preprocessed.cache();
		}
		
		// learn the features and save them to a file
		features = learnFeatures(input1Preprocessed);
		if (saveModel == true) {
			learn.saveToFile(features, pathPrefix + "_features_layer_" + layerIndex + "_" + System.currentTimeMillis(), sparkContext);
		}
		
		// perform feature extraction
		JavaRDD<Tuple2<Vector, Vector>> represent = extractFeatures(input2, features);
		
		// perform pooling
		JavaRDD<Tuple2<Vector, Vector>> pooled = pool(represent).repartition(numPartitions).cache();
		
		// we are not in the last layer, therefore we force the previous computation
		if (notLast) {
			pooled.count();
		}
		
		return pooled;
	}

	
	/**
	 * Overriden method the performs a test pass through the data for the current layer.
	 * 
	 * @param data Input distributed dataset for the current layer.
	 * @param featFile Array of input files that contain saved parameters of the trained model.
	 * What is the convention here?
	 * @throws Exception Standard Exception object.
	 */
    @Override
    public JavaRDD<Tuple2<Vector, Vector>> test(JavaRDD<Tuple2<Vector, Vector>> data, String[] featFile) throws Exception {
    	
    	// TODO:: Make this more automatic!
    	int numPartitions = 400 * 4; 	// num-workers * cores_per_worker * succesive tasks

    	// setup the preprocessor
    	if (preprocess != null) {
    		preprocess.loadFromFile(featFile, sparkContext); //this needs to be changed
    	}
    	
    	// load the features from file
    	Vector[] features = learn.loadFromFile(featFile[layerIndex+2], sparkContext);
    	QuickSortVector.quickSort(features, 0, features.length-1);
    	
    	// perform feature extraction
    	JavaRDD<Tuple2<Vector, Vector>> represent = extractFeatures(data, features);
    	
    	// perform pooling
    	JavaRDD<Tuple2<Vector, Vector>> pooled = pool(represent).repartition(numPartitions).cache();
    	
    	return pooled;
    }
   
    
    /**
     * Return the path prefix for saving files.
     * 
     * @return Path prefix.
     */
    public String getPathPrefix() {
    	return pathPrefix;
    }
    
    
    /**
     * Set the path prefix parameter for saving files.
     * 
     * @param s Input path prefix.
     */
    public void setPathPrefix(String s) {
    	pathPrefix = s;
    }
    
    
    /**
     * Return the layer index.
     * 
     * @return Layer index.
     */
    public int getLayerIndex() {
    	return layerIndex;
    }
    
    
    /**
     * Set the layer index.
     * 
     * @param l Input layer index.
     */
    public void setLayerIndex(int l) {
    	layerIndex = l;
    }
    
    
    /**
     * Set the spark context for saving files.
     * 
     * @param sc Input spark context.
     */
    @Override
    public void setSparkContext(JavaSparkContext sc) {
    	sparkContext = sc;
    }
    
    
    /**
     * Set the saveModel parameter, which indicates saving or not.
     * 
     * @param saveModel Input boolean.
     */
    @Override
	public void setSaveModel(boolean saveModel) {
		this.saveModel = saveModel;
	}
	
	
	/**
	 * Return the boolean that indicates saving.
	 * 
	 * @return Boolean value.
	 */
    @Override
	public boolean getSaveModel() {
		return saveModel;
	}
	
	
	/**
	 * Sets the notLast boolean, indicating if we are in the last layer.
	 * 
	 * @param notLast Indicator for last layer.
	 */
    @Override
	public void setNotLast(boolean notLast) {
		this.notLast = notLast;
	}
}
