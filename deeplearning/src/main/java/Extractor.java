package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;

/**
 * All Extractors (Matrix multiply, Convolution,..) should implement this interface. See DummyExtractor for example.
 * 
 * @author Arttu Voutilainen
 *
 */
public interface Extractor extends Function<Vector, Vector> {

	/**
	 * Set BaseLayer configuration to be used by this extractor.
	 * 
	 * @param configLayer The configuration for the BaseLayer
	 */
	//public void setConfigLayer(ConfigBaseLayer configLayer);
	
	
	/**
	 * Set features to be used by this extractor.
	 *
	 * @param features The features this extractor will use
	 */
	public void setFeatures(Vector[] features);

	
	/**
	 * Main method that is called by passing it to a map call. 
	 * This method will be applied to each data point independently.
	 * Here, 2-d convolution will be implmented based on FFT.
	 *
	 * @param data Vector representing one data point
	 * @return A new representation of the input after applying the feature extraction
	**/
	public Vector call(Vector data) throws Exception;

}
