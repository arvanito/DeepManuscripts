package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigFeatureExtractor;

import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;


/**
 * Class for feature extraction using multiplication for the first layer of learning.
 * Extraction of overlapping patches and multiplication. Equivalent to convolution.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */

public class ConvMultiplyExtractor implements Extractor {

	private static final long serialVersionUID = 7991635895652585866L;

	private DenseMatrix zca;
	private DenseVector mean;
	private Vector[] features;	// array of learned feature Vectors

	private int inputRows;
	private int inputCols;
	private int featureRows;
	private int featureCols;
	//private int validRows;
	//private int validCols;

	private ConfigFeatureExtractor.NonLinearity nonLinearity = null;
	private double alpha; // non-linearity (threshold)

	/**
	 * Constructor 
	 * @param configLayer The input configuration for the current layer
	 * @param preProcess The input PreProcess configuration
	 */
	public ConvMultiplyExtractor(ConfigBaseLayer configLayer) {
		setConfig(configLayer);
	}


	/**
	 * Constructor 
	 * @param configLayer The input configuration for the current layer
	 * @param preProcess The input PreProcess configuration
	 * @param features The input feature learned from the previous step
	 */
	public ConvMultiplyExtractor(ConfigBaseLayer configLayer, PreProcessZCA preProcess, Vector[] features) {
		setConfig(configLayer);
		this.features = features;
	}
	
	/**
	 * Setter method for the ConfigBaseLayer object.
	 * 
	 * @param configLayer The ConfigBaseLayer object
	 */
	private void setConfig(ConfigBaseLayer configLayer) {
		ConfigFeatureExtractor conf = configLayer.getConfigFeatureExtractor();
		inputCols = conf.getInputDim1();
		inputRows = conf.getInputDim2();
		if(inputCols == 0) throw new RuntimeException("Configured input dimension 1 is 0");
		if(inputCols == 0) throw new RuntimeException("Configured input dimension 2 is 0");

		featureCols = conf.getFeatureDim1();
		featureRows = conf.getFeatureDim2();
		if(featureCols == 0 || featureCols > inputCols) throw new RuntimeException("Configured feature dimension 1 is 0 or > input dimension 1");
		if(featureRows == 0 || featureRows > inputRows) throw new RuntimeException("Configured feature dimension 2 is 0 or > input dimension 2");

		//validRows = inputRows - featureRows + 1;
		//validCols = inputCols - featureCols + 1;
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
		
		/** Get necessary parameters for the feature extraction process **/
		
		// number of features learned
		/*int numFeatures = 0;
		if (configLayer.hasConfigKmeans()) {
			numFeatures = configLayer.getConfigKmeans().getNumberOfClusters();
		} else if (configLayer.hasConfigAutoencoders()) {
			numFeatures = configLayer.getConfigAutoencoders().getNumberOfUnits();
		}*/
		
		// filters, convert from Vector[] to DenseMatrix
		DenseMatrix D = MatrixOps.convertVectors2Mat(features);

		// reshape data vector to a matrix and extract all overlapping patches
		int[] dims = {inputRows, inputCols};
		int[] rfSize = {featureRows, featureCols};
		DenseMatrix M = MatrixOps.reshapeVec2Mat((DenseVector) data, dims);	
		DenseMatrix patches = MatrixOps.im2colT(M, rfSize);
		
		// allocate memory for the output vector
		DenseMatrix out = new DenseMatrix(patches.numRows(),D.numRows(),new double[patches.numRows()*D.numRows()]);	
		DenseMatrix patchesOut = new DenseMatrix(patches.numRows(),patches.numCols(),new double[patches.numRows()*patches.numCols()]);
		
		// get necessary data from the PreProcessor
		if (zca != null && mean != null) {

			// epsilon for pre-processing
			//double eps1 = configLayer.getConfigPreprocess().getEps1();
			
			// preprocess the data point with contrast normalization and ZCA whitening
			//patches = MatrixOps.localMatContrastNorm(patches, eps1);
			patches = MatrixOps.localMatSubtractMean(patches, mean);
			
			//patches = patches.multiply(zca);
			BLAS.gemm(1.0, patches, zca, 0.0, patchesOut);
		} else {
			patchesOut = patches;
		}
	
		// multiply the matrix of the learned features with the preprocessed data point
		BLAS.gemm(1.0, patchesOut, D.transpose(), 0.0, out);
		DenseVector outVec = MatrixOps.reshapeMat2Vec(out);
		
		// apply non-linearity
		outVec = MatrixOps.applyNonLinearityVec(outVec, nonLinearity, alpha);
		
		return outVec;
	}
	
}
