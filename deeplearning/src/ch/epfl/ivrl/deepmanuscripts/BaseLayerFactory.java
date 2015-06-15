package ch.epfl.ivrl.deepmanuscripts;


import ch.epfl.ivrl.deepmanuscripts.BaseLayer;
import ch.epfl.ivrl.deepmanuscripts.DeepModelSettings.ConfigBaseLayer;
import ch.epfl.ivrl.deepmanuscripts.DeepModelSettings.ConfigPooler;

/**
 * 
 * A Factory class for BaseLayers. Given the configuration for a base layer,
 * the class is responsible for properly instantiating  a derived object of type BaseLayer.  
 * 
 * @author Viviana Petrescu
 *
 */

public class BaseLayerFactory {

	/**
	 * Main method that creates a BaseLayer from the loaded protocol buffer configuration.
	 * 
	 * @param configLayer ConfigBaseLayer object read from the protocol buffer file.
	 * @param layerIndex Index that represents the number of layer created.
	 * @param pathPrefix Path prefix for saving the trained model.
	 * @return BaseLayer object that represents the current layer.
	 */
	static public BaseLayer createBaseLayer(ConfigBaseLayer configLayer, int layerIndex, String pathPrefix) {
		
		// set up the preprocessor, optional for every layer
		PreProcessZCA preprocessor = null;
		if (configLayer.hasConfigPreprocess()) {
			preprocessor = new PreProcessZCA(configLayer);
		}
		
		// assert one of this two needs to be true. We should have either K-means or Auto-encoders
		assert(configLayer.hasConfigAutoencoders() || configLayer.hasConfigKmeans());
		
		// set up the learner, required for every layer
		Learner learner = null;
		if (configLayer.hasConfigKmeans()) {
			learner = new KMeansLearner(configLayer);
		}
		// TODO:: Put Autoencoder classes from master branch
		if (configLayer.hasConfigAutoencoders()) {
			//learner = new AutoencoderLearner(configLayer);	
		}
		
		// set up the extractor, depending on the layer 
		// either FFT convolutional extraction, or matrix-vector multiplication extraction
		Extractor extractor = null;
		if (configLayer.hasConfigFeatureExtractor()) {
			// we use the ConvMultiplyExtractor for contrast normalized patches
			extractor = new ConvMultiplyExtractor(configLayer);
			
			// otherwise, we use the FFTConvolutionExtractor
			//extractor = new FFTConvolutionExtractor(configLayer);
		} else {
			extractor = new MultiplyExtractor(configLayer);
		}
		
		// set up the pooler, by default max-pooler
		// TODO:: Incorporate an average pooler
		Pooler pooler = null;
		if (configLayer.getConfigPooler().getPoolType() == ConfigPooler.PoolType.MAX) {
			pooler = new MaxPoolerExtended(configLayer);
		}
 
		// create and return the base layer, setting the current layer index
		BaseLayer b = new BaseLayer(configLayer, preprocessor, learner, extractor, pooler, layerIndex);
		
		// set the path prefix for saving the files, using the layer index as indicator
		// if the path prefix is not empty, then we save the model
		if (pathPrefix.length() == 0) {
			b.setPathPrefix("");
		} else {
			b.setPathPrefix(pathPrefix + Integer.toString(layerIndex));
			b.setSaveModel(true);
		}
		
		return b;
	}
	
}
