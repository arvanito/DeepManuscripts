package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;


/**
 * Class for feature extraction using matrix multiplications. 
 * 
 * @author Nikolaos Arvanitopoulos
 */
public class MultiplyExtractor implements Extractor {
	
	private static final long serialVersionUID = -6353736803330058842L;

	private ConfigBaseLayer configLayer;	// layer configuration from the protocol buffer
	private PreProcessZCA preProcess; 		// pre-processing information 
	private Vector[] features;				// array of learned feature Vectors
	
	
	/**
	 * Constructor 
	 * @param configLayer The input configuration for the current layer
	 * @param preProcess The input PreProcess configuration
	 * @param features The input feature learned from the previous step
	 */
	public MultiplyExtractor(ConfigBaseLayer configLayer, PreProcessZCA preProcess, Vector[] features) {
		this.configLayer = configLayer;
		this.preProcess = preProcess;
		this.features = features;
	}
	
	
	/**
	 * Getter method for the ConfigBaseLayer object.
	 * 
	 * @return The ConfigBaseLayer object
	 */
	public ConfigBaseLayer getConfigLayer() {
		return configLayer;
	}
	
	
	/**
	 * Getter method for the PreProcessor object.
	 * 
	 * @return The PreProcessor object
	 */
	public PreProcessZCA getPreProcess() {
		return preProcess;
	}
	
	
	/**
	 * Getter method for the learned features.
	 * 
	 * @return The learned features
	 */
	public Vector[] getFeatures() {
		return features;
	}
	
	
	/**
	 * Setter method for the ConfigBaseLayer object.
	 * 
	 * @param configLayer The ConfigBaseLayer object
	 */
	@Override
	public void setConfigLayer(ConfigBaseLayer configLayer) {
		this.configLayer = configLayer;
	}
	
	
	/**
	 * Setter method for the PreProcessor object.
	 * 
	 * @param preProcess The PreProcessor object
	 */
	public void setPreProcess(PreProcessZCA preProcess) {
		this.preProcess = preProcess;
	}
	
	
	/**
	 * Setter method for learned features.
	 * 
	 * @param features Input learned features
	 */
	@Override
	public void setFeatures(Vector[] features) {
		this.features = features;
	}
	
	
	/**
	 * Method that is called during a map call.
	 * 
	 * @param data Input Vector
	 * @return Extracted feature
	 */
	@Override
	public Vector call(Vector data) throws Exception {
		
		// number of features learned
		/*int numFeatures = 0;
		if (configLayer.hasConfigKmeans()) {
			numFeatures = configLayer.getConfigKmeans().getNumberOfClusters();
		} else if (configLayer.hasConfigAutoencoders()) {
			numFeatures = configLayer.getConfigAutoencoders().getNumberOfUnits();
		}*/
		
		// filters, convert from Vector[] to DenseMatrix
		DenseMatrix D = MatrixOps.convertVectors2Mat(features);

		// get necessary data from the PreProcessor
		DenseVector dataDense = (DenseVector) data;
		if (configLayer.hasConfigPreprocess()) {
			// ZCA Matrix
			DenseMatrix zca = preProcess.getZCA();

			// mean from ZCA
			DenseVector zcaMean = preProcess.getMean();

			// epsilon for pre-processing
			double eps1 = configLayer.getConfigPreprocess().getEps1();
			
			// preprocess the data point with contrast normalization and ZCA whitening
			dataDense = MatrixOps.localVecContrastNorm(dataDense, eps1);
			dataDense = MatrixOps.localVecSubtractMean(dataDense, zcaMean);
			BLAS.gemv(true, 1.0, zca, dataDense, 0.0, dataDense);
		}
	
		// multiply the matrix of the learned features with the preprocessed data point
		BLAS.gemv(false, 1.0, D, dataDense, 0.0, dataDense);
				
		return dataDense;
	}

}