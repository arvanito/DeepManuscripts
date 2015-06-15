package ch.epfl.ivrl.deepmanuscripts;

import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import ch.epfl.ivrl.deepmanuscripts.DeepModelSettings.ConfigBaseLayer;
import ch.epfl.ivrl.deepmanuscripts.DeepModelSettings.ConfigFeatureExtractor;
import scala.Tuple2;


/**
 * Class for feature extraction using multiplication for the first layer of learning.
 * Extraction of overlapping patches and multiplication. Equivalent to convolution.
 * It is used instead of FFTConvolution, because of the contrast normalization pre-processing 
 * for the input patches.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class ConvMultiplyExtractor implements Extractor {

	private static final long serialVersionUID = 7991635895652585866L;

	private DenseVector mean;
	private DenseMatrix zca;
	private Vector[] features;

	private int inputRows;
	private int inputCols;
	private int featureRows;
	private int featureCols;
	
	private double eps1;

	private ConfigFeatureExtractor.NonLinearity nonLinearity = null;
	private double alpha; // non-linearity (threshold)

	
	/**
	 * Constructor that sets the current base layer configuration. 
	 * 
	 * @param configLayer The current base layer configuration.
	 */
	public ConvMultiplyExtractor(ConfigBaseLayer configLayer) {
		setConfig(configLayer);
	}
	
	
	/**
	 * Method that sets the current base layer configuration.
	 * 
	 * @param configLayer The ConfigBaseLayer object.
	 */
	private void setConfig(ConfigBaseLayer configLayer) {
		
		ConfigFeatureExtractor conf = configLayer.getConfigFeatureExtractor();
		
		// get input data dimensions
		inputCols = conf.getInputDim1();
		inputRows = conf.getInputDim2();
		if(inputCols == 0) throw new RuntimeException("Configured input dimension 1 is 0");
		if(inputCols == 0) throw new RuntimeException("Configured input dimension 2 is 0");

		// get input feature dimensions
		featureCols = conf.getFeatureDim1();
		featureRows = conf.getFeatureDim2();
		if(featureCols == 0 || featureCols > inputCols) throw new RuntimeException("Configured feature dimension 1 is 0 or > input dimension 1");
		if(featureRows == 0 || featureRows > inputRows) throw new RuntimeException("Configured feature dimension 2 is 0 or > input dimension 2");

		// get non-linearity
		nonLinearity = conf.getNonLinearity();
		
		// get the alpha threshold, in case of threshold non-linearity
		if (conf.hasSoftThreshold()) {
			alpha = conf.getSoftThreshold();
		}
	}
	
	
	/**
	 * Method that sets the pre-processing mean and ZCA variables.
	 * 
	 * @param mean Input mean vector.
	 * @param zca Input ZCA matrix.
	 */
	@Override
	public void setPreProcessZCA(DenseVector mean, DenseMatrix zca) {
		this.mean = mean;
		this.zca = zca;
	}
	
	
	/**
	 * Method that sets the learned features.
	 * 
	 * @param features Input learned features.
	 */
	@Override
	public void setFeatures(Vector[] features) {
		this.features = features;
	}
	
	
	/**
	 * Method that set the epsilon parameter for contrast normalization.
	 * 
	 * @param eps1 Variable for contrast normalization.
	 */
	@Override
	public void setEps1(double eps1) {
		this.eps1 = eps1;
	}
	
	
	/**
	 * Main method that performs convolutional feature extraction.
	 * 
	 * @param pair Input tuple.
	 * @return New representation of the input tuple.
	 */
	@Override
	public Tuple2<Vector, Vector> call(Tuple2<Vector, Vector> pair) {
		
		// get the data part of the tuple
		Vector data = pair._2;
		
		// convert the features from Vector[] to DenseMatrix
		DenseMatrix D = MatrixOps.convertVectors2Mat(features);

		// reshape the input data vector to a matrix and extract all overlapping patches
		int[] dims = {inputRows, inputCols};
		int[] rfSize = {featureRows, featureCols};
		DenseMatrix M = MatrixOps.reshapeVec2Mat((DenseVector) data, dims);	
		DenseMatrix patches = MatrixOps.im2colT(M, rfSize);
		
		// allocate memory for the output vector
		DenseMatrix out = new DenseMatrix(patches.numRows(),D.numRows(),new double[patches.numRows()*D.numRows()]);	
		DenseMatrix patchesOut = new DenseMatrix(patches.numRows(),patches.numCols(),new double[patches.numRows()*patches.numCols()]);
		
		// do the feature extraction
		if (zca != null && mean != null) {
			
			// preprocess the data point with contrast normalization and ZCA whitening
			patches = MatrixOps.localMatContrastNorm(patches, eps1);
			patches = MatrixOps.localMatSubtractMean(patches, mean);
			
			BLAS.gemm(1.0, patches, zca, 0.0, patchesOut);
		} else {
			patchesOut = patches;
		}
	
		// multiply the matrix of the learned features with the pre-processed data point
		BLAS.gemm(1.0, patchesOut, D.transpose(), 0.0, out);
		DenseVector outVec = MatrixOps.reshapeMat2Vec(out);
		
		// apply non-linearity
		outVec = MatrixOps.applyNonLinearityVec(outVec, nonLinearity, alpha);
		
		return new Tuple2<Vector, Vector>(pair._1, outVec);
	}
	
}
