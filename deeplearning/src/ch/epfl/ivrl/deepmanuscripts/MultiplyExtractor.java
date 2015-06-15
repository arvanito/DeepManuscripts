package ch.epfl.ivrl.deepmanuscripts;

import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import ch.epfl.ivrl.deepmanuscripts.DeepModelSettings.ConfigBaseLayer;
import ch.epfl.ivrl.deepmanuscripts.DeepModelSettings.ConfigFeatureExtractor;
import scala.Tuple2;


/**
 * Class for feature extraction using matrix multiplications. 
 * 
 * @author Nikolaos Arvanitopoulos
 */
public class MultiplyExtractor implements Extractor {
	
	private static final long serialVersionUID = -6353736803330058842L;

	private DenseMatrix zca;
	private DenseVector mean;
	private Vector[] features;				
	
	private double eps1;
	
	private ConfigFeatureExtractor.NonLinearity nonLinearity = null; 	
	private double alpha;												
	
	
	/**
	 * Constructor that sets the current base layer configuration. 
	 * 
	 * @param configLayer The current base layer configuration.
	 */
	public MultiplyExtractor(ConfigBaseLayer configLayer) {
		setConfig(configLayer);
	}
	
	
	/**
	 * Method that sets the current base layer configuration.
	 * 
	 * @param configLayer The ConfigBaseLayer object.
	 */
	private void setConfig(ConfigBaseLayer configLayer) {
		
		ConfigFeatureExtractor conf = configLayer.getConfigFeatureExtractor();
		
		// set the non-linearity 
		nonLinearity = conf.getNonLinearity();
		
		// set the alpha parameter in case of threshold non-linearity
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
	 * Main method that performs feature extraction through matrix multiplications.
	 * 
	 * @param pair Input tuple.
	 * @return New representation of the input tuple.
	 */
	@Override
	public Tuple2<Vector, Vector> call(Tuple2<Vector, Vector> pair) {
		
		// get the data part of the tuple
		Vector data = pair._2;
		DenseVector dataDense = (DenseVector) data;
		
		// convert the features from Vector[] to DenseMatrix
		DenseMatrix D = MatrixOps.convertVectors2Mat(features);
		
		// allocate memory for the output vector
		DenseVector dataOut = new DenseVector(new double[D.numRows()]);
		DenseVector dataDenseOut = new DenseVector(new double[dataDense.size()]);
		
		// most probably we do not need any pre-processing here, 
		// data is already whitened
		if (zca != null && mean != null) {
			
			// preprocess the data point with contrast normalization and ZCA whitening
			dataDense = MatrixOps.localVecContrastNorm(dataDense, eps1);
			dataDense = MatrixOps.localVecSubtractMean(dataDense, mean);
			
			BLAS.gemv(1.0, zca.transpose(), dataDense, 0.0, dataDenseOut);
		} else {
			dataDenseOut = dataDense;
		}
	
		// multiply the matrix of the learned features with the pre-processed data point
		BLAS.gemv(1.0, D, dataDenseOut, 0.0, dataOut);
		
		// apply non-linearity
		if (nonLinearity != null) {
			dataOut = MatrixOps.applyNonLinearityVec(dataOut, nonLinearity, alpha);
		}
		
		return new Tuple2<Vector, Vector>(pair._1, dataOut);
	}

}
