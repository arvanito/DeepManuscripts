package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;

/**
 * Class that performs ZCA whitening on the dataset.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class PreProcessZCA implements PreProcessor {

	private static final long serialVersionUID = 2534694814224730024L;

	private ConfigBaseLayer configLayer;
	private DenseVector mean;
	private DenseMatrix ZCA;
	
	// checks if we are doing convolutional preprocessing or not
	// TODO: Initialize it somewhere!
	boolean conv;
	
	public PreProcessZCA() {}
	public PreProcessZCA(ConfigBaseLayer c) {
		setConfig(c);
	}
	public PreProcessZCA(DenseVector mean, DenseMatrix ZCA, ConfigBaseLayer conf) {
		setConfig(conf);
		this.mean = mean;
		this.ZCA = ZCA;
	}
	
	/**
	 * Getter method for the mean vector.
	 * 
	 * @return The mean vector
	 */
	public DenseVector getMean() {
		return mean;
	}
	
	
	/**
	 * Getter for the ZCA matrix.
	 * 
	 * @return The ZCA matrix
	 */
	public DenseMatrix getZCA() {
		return ZCA;
	}
	
	
	/**
	 * Method that sets the layer configuration.
	 * 
	 * @param configLayer The layer configuration object
	 */
	private void setConfig(ConfigBaseLayer configLayer) {
		this.configLayer = configLayer;
		if (this.configLayer.hasConfigPreprocess()) {
			conv = true;
		} else {
			conv = false;
		}
	}
	
	
	/**
	 * Setter for the mean vector.
	 * 
	 * @param mean Mean vector
	 */
	private void setMean(DenseVector mean) {
		this.mean = mean;
	}
	
	
	/**
	 * Setter for the ZCA matrix.
	 * 
	 * @param ZCA The ZCA matrix
	 */
	private void setZCA(DenseMatrix ZCA) {
		this.ZCA = ZCA;
	}
	
	
	/**
	 * Method that performs ZCA whitening on the data.
	 * 
	 * @param mat Input distributed RowMatrix
	 * @param e Parameter for eigenvalue normalization
	 * @return ZCA matrix of type DenseMatrix
	 */
	private DenseMatrix performZCA(RowMatrix mat, double e) {
		
		// compute SVD of the data Matrix
		// the right singular Vectors are the eigenvectors of the covariance, do the integer casting here!!!
		SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD((int) mat.numCols(), true, 1.0E-9d);
		DenseMatrix V = (DenseMatrix) svd.V();		// right singular Vectors
		DenseVector s = (DenseVector) svd.s();		// singular values
		
		// the eigenvalues of the covariance are the squares of the singular values
		// add a regularizer and compute the square root
		long n = mat.numRows();	
		int ss = s.size();
		double[] l = new double[ss];
		for (int i = 0; i < ss; i++) {
			l[i] = (s.apply(i) * s.apply(i)) / (n - 1);
			l[i] = 1.0 / Math.sqrt(l[i] + e);
		}

		// compute the ZCA matrix by V * Lambda * V'
		DenseMatrix ZCA = new DenseMatrix(ss, ss, new double[ss*ss]);
		DenseMatrix ZCAout = new DenseMatrix(ss, ss, new double[ss*ss]);
		BLAS.gemm(1.0, Matrices.diag(Vectors.dense(l)), V.transpose(), 0.0, ZCA);
		//ZCA = V.multiply(ZCA);
		BLAS.gemm(1.0, V, ZCA, 0.0, ZCAout);
		
		return ZCAout;
	}
	
	
	/**
	 * Main method that preprocesses the dataset. 
	 * 
	 * @param data Input distributed dataset
	 * @param configLayer Current layer configuration from the protocol buffer
	 * @return Preprocessed distributed dataset
	 */
	@Override
	public JavaRDD<Vector> preprocessData(JavaRDD<Vector> data) {

		// assign eps1 for pre-processing
		double eps1 = configLayer.getConfigPreprocess().getEps1();

		// apply contrast normalization
		data = data.map(new ContrastNormalization(eps1));

		// convert the JavaRRD<Vector> to a distributed RowMatrix (through Scala RDD<Vector>)
		RowMatrix rowData = new RowMatrix(data.rdd());

		// compute mean data Vector
		MultivariateStatisticalSummary summary = rowData.computeColumnSummaryStatistics();
		DenseVector m = (DenseVector) summary.mean();
		setMean(m);

		// remove the mean from the dataset
		data = data.map(new SubtractMean(m));	

		// create distributed Matrix from centralized data, input to ZCA
		rowData = new RowMatrix(data.rdd());
		
		// perform ZCA whitening and project the data to decorrelate them
		DenseMatrix ZCA = performZCA(rowData, configLayer.getConfigPreprocess().getEps2());
		rowData = rowData.multiply(ZCA);
		setZCA(ZCA);
			
		// convert the distributed RowMatrix into a JavaRDD<Vector> 
		data = new JavaRDD<Vector>(rowData.rows(), data.classTag());

		return data;
	}
	
	
	/**
	 * Method that preprocesses input data with the learned mean vector and ZCA matrix.
	 * 
	 * @param data Input data in Vector format
	 * @return Preprocessed output
	 */
	@Override
	public Vector call(Vector data) {		
		
		DenseVector dataDense = (DenseVector) data;
		
		// epsilon for pre-processing
		double eps1 = configLayer.getConfigPreprocess().getEps1();
		
		DenseVector outVec = new DenseVector(new double[data.size()]);
		
		// preprocess data depending on the conv flag
		if (conv == false) {
			// preprocess the data point with contrast normalization and ZCA whitening
			dataDense = MatrixOps.localVecContrastNorm(dataDense, eps1);
			dataDense = MatrixOps.localVecSubtractMean(dataDense, mean);
			BLAS.gemv(1.0, ZCA.transpose(), dataDense, 0.0, outVec);
		} else {
			// reshape data vector to a matrix and extract all overlapping patches
			int[] dims = {configLayer.getConfigFeatureExtractor().getInputDim1(), configLayer.getConfigFeatureExtractor().getInputDim2()};
			int[] rfSize = {configLayer.getConfigFeatureExtractor().getFeatureDim1(), configLayer.getConfigFeatureExtractor().getFeatureDim2()};
			DenseMatrix M = MatrixOps.reshapeVec2Mat((DenseVector) data, dims);	
			DenseMatrix patches = MatrixOps.im2colT(M, rfSize);
		
			// preprocess the data point with contrast normalization and ZCA whitening
			patches = MatrixOps.localMatContrastNorm(patches, eps1);
			patches = MatrixOps.localMatSubtractMean(patches, mean);
			
			DenseMatrix dataOut = new DenseMatrix(patches.numRows(),ZCA.numCols(),new double[patches.numRows()*ZCA.numCols()]);
			BLAS.gemm(1.0, patches, ZCA, 0.0, dataOut);
			outVec = MatrixOps.reshapeMat2Vec(dataOut);
		}
		
		return outVec;
	}
	
	/**
	 *  Sets up the preprocessor. It loads the saved weights from the disk.
	 * @param filename
	 **/
	public void loadFromFile(String filename, JavaSparkContext sc) {
		//NOTE Since ZCA and mean are both expected to be small (max 64x64)
		// their loading/saving should not be a bottleneck
		mean = (DenseVector) LinAlgebraIOUtils.loadVectorFromObject(filename+"_mean", sc);
		ZCA = (DenseMatrix) LinAlgebraIOUtils.loadMatrixFromObject(filename + "_zca", sc);
		//this part was used for training of second layer from first layer
//		mean = (DenseVector) LinAlgebraIOUtils.loadVectorFromObject("testDeep26000filters_x_0_preprocess1431876315796_mean", sc);
//		ZCA = (DenseMatrix) LinAlgebraIOUtils.loadMatrixFromObject("testDeep26000filters_x_0_preprocess1431876315796_zca", sc);
	}
	
	/**
	 *  Saves the fields necessary to reconstruct a preprocessor object. 
	 *  Depending on the preprocessor type, more than one file will be saved.
	 * @param filename common base filename path at which a suffix is added for every field
	 * 					that is saved
	 **/
	public void saveToFile(String filename, JavaSparkContext sc) {
		//TODO
		//NOTE Since ZCA and mean are both expected to be small (max 64x64)
		// their loading/saving should not be a bottleneck
		LinAlgebraIOUtils.saveVectorToObject(this.mean, filename + "_mean", sc);
		LinAlgebraIOUtils.saveMatrixToObject(this.ZCA, filename + "_zca", sc);
	}
}
