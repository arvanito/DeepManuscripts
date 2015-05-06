package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigFeatureExtractor;

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

	private ConfigFeatureExtractor.NonLinearity nonLinearity = null; 	// nonLinearity, by default NONE
	private double alpha;												// non-linearity optional threshold
	private DenseMatrix zca;
	private DenseVector mean;
	private Vector[] features;				// array of learned feature Vectors
	
	
	public MultiplyExtractor() {}
	
	/**
	 * Constructor 
	 * @param configLayer The input configuration for the current layer
	 * @param preProcess The input PreProcess configuration
	 */
	public MultiplyExtractor(ConfigBaseLayer configLayer) {
		setConfigLayer(configLayer);
	}
	
	@Override
	public void setConfigLayer(ConfigBaseLayer configLayer) {
		ConfigFeatureExtractor conf = configLayer.getConfigFeatureExtractor();
		nonLinearity = conf.getNonLinearity();
		System.out.println(nonLinearity);
		if (conf.hasSoftThreshold()) {
			alpha = conf.getSoftThreshold();
		}
	}
	
	@Override
	public void setPreProcessZCA(DenseMatrix zca, DenseVector mean) {
		this.zca = zca;
		this.mean = mean;
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
		// filters, convert from Vector[] to DenseMatrix
		DenseMatrix D = MatrixOps.convertVectors2Mat(features);

		// get necessary data from the PreProcessor
		DenseVector dataDense = (DenseVector) data;
		
		// allocate memory for the output vector
		DenseVector dataOut = new DenseVector(new double[D.numRows()]);
		DenseVector dataDenseOut = new DenseVector(new double[dataDense.size()]);
		
		// most probably we do not need any pre-processing here, 
		// data is already whitened
		if (zca != null && mean != null) {
			
			// epsilon for pre-processing
			//double eps1 = configLayer.getConfigPreprocess().getEps1();
			
			// preprocess the data point with contrast normalization and ZCA whitening
			//dataDense = MatrixOps.localVecContrastNorm(dataDense, eps1);
			dataDense = MatrixOps.localVecSubtractMean(dataDense, mean);
			//dataDense = zca.transpose().multiply(dataDense);
			BLAS.gemv(1.0, zca.transpose(), dataDense, 0.0, dataDenseOut);
		} else {
			dataDenseOut = dataDense;
		}
	
		// multiply the matrix of the learned features with the preprocessed data point
		BLAS.gemv(1.0, D, dataDenseOut, 0.0, dataOut);
		
		// apply non-linearity
		if (nonLinearity != null) {
			dataOut = MatrixOps.applyNonLinearityVec(dataOut, nonLinearity, alpha);
		}
		
		return dataOut;
	}
}
