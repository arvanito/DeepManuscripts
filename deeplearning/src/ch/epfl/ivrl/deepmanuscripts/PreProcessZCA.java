package ch.epfl.ivrl.deepmanuscripts;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
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

import ch.epfl.ivrl.deepmanuscripts.DeepModelSettings.ConfigBaseLayer;
import scala.Tuple2;

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
	boolean conv;
	
	
	/**
	 * Empty Constructor.
	 */
	public PreProcessZCA() {}
	
	
	/**
	 * Constructor that set the base layer configuration.
	 * 
	 * @param c Base layer configuration.
	 */
	public PreProcessZCA(ConfigBaseLayer c) {
		configLayer = c;
	}
	
	
	/**
	 * Constructor that sets the mean and ZCA variables.
	 * 
	 * @param mean Mean vector.
	 * @param ZCA ZCA matrix.
	 */
	public PreProcessZCA(DenseVector mean, DenseMatrix ZCA) {
		this.mean = mean;
		this.ZCA = ZCA;
	}
	
	
	/**
	 * Method that returns the ConfigBaseLayer object.
	 * 
	 * @return The ConfigBaseLayer object.
	 */
	public ConfigBaseLayer getConfigLayer() {
		return configLayer;
	}
	
	
	/**
	 * Method that returns the mean vector.
	 * 
	 * @return The mean vector.
	 */
	public DenseVector getMean() {
		return mean;
	}
	
	
	/**
	 * Method that returns the ZCA matrix.
	 * 
	 * @return The ZCA matrix.
	 */
	public DenseMatrix getZCA() {
		return ZCA;
	}
	
	
	/**
	 * Method that sets the layer configuration.
	 * 
	 * @param configLayer The layer configuration object.
	 */
	@Override
	public void setConfigLayer(ConfigBaseLayer configLayer) {
		this.configLayer = configLayer;
		if (this.configLayer.hasConfigPreprocess()) {
			conv = true;
		} else {
			conv = false;
		}
	}
	
	
	/**
	 * Method that sets the mean vector.
	 * 
	 * @param mean Mean vector.
	 */
	public void setMean(DenseVector mean) {
		this.mean = mean;
	}
	
	
	/**
	 * Method that sets the ZCA matrix.
	 * 
	 * @param ZCA The ZCA matrix.
	 */
	public void setZCA(DenseMatrix ZCA) {
		this.ZCA = ZCA;
	}
	
	
	/**
	 * Method that performs ZCA whitening on the data.
	 * 
	 * @param mat Input distributed RowMatrix.
	 * @param e Parameter for eigenvalue normalization.
	 * @return ZCA matrix of type DenseMatrix.
	 */
	public DenseMatrix performZCA(RowMatrix mat, double e) {
		
		// compute SVD of the data Matrix
		// the right singular Vectors are the eigenvectors of the covariance
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
		BLAS.gemm(1.0, V, ZCA, 0.0, ZCAout);
		
		return ZCAout;
	}
	
	
	/**
	 * Main method that preprocesses the dataset. 
	 * 
	 * @param pairData Input distributed dataset.
	 * @return Preprocessed distributed dataset.
	 */
	@Override
	public JavaRDD<Tuple2<Vector, Vector>> preprocessData(JavaRDD<Tuple2<Vector, Vector>> pairData) {

		// extract data part from the Tuple RDD
		JavaRDD<Vector> data = pairData.map(
					new Function<Tuple2<Vector, Vector>, Vector>() {
						private static final long serialVersionUID = 7953912605428885035L;

						public Vector call(Tuple2<Vector, Vector> pair) {
							return pair._2;
						}
					}
				);

		// apply contrast normalization
		data = data.map(new ContrastNormalization(configLayer.getConfigPreprocess().getEps1()));

		// convert the JavaRRD<Vector> to a distributed RowMatrix
		RowMatrix rowData = new RowMatrix(data.rdd());

		// compute mean data Vector
		MultivariateStatisticalSummary summary = rowData.computeColumnSummaryStatistics();
		DenseVector m = (DenseVector) summary.mean();
		setMean(m);

		// remove the mean from the dataset
		data = data.map(new SubtractMean(m));	

		// create distributed Matrix from centralized data, input to ZCA
		rowData = new RowMatrix(data.rdd());
		
		// perform ZCA whitening and project the data to
		DenseMatrix ZCA = performZCA(rowData, configLayer.getConfigPreprocess().getEps2());
		setZCA(ZCA);
			
		// create a JavaRDD<Tuple2<Vector, Vector>> with the whitened data
		pairData = pairData.map(new convert2Tuple2(m, ZCA));
		
		return pairData;
	}
	
	
	/**
	 * Method that preprocesses input data with the learned mean vector and ZCA matrix.
	 * It is not needed!!!
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
	 * Method that loads the pre-processing parameters from files.
	 * First file name is the mean vector, second file name is the ZCA matrix.
	 *  
	 * @param pathPrefix Path prefix to load the pre-processing variables.
	 * @param sc Spark context variable.
	 **/
	@Override
	public void loadFromFile(String pathPrefix, JavaSparkContext sc) {

		// first file name is the mean
		mean = (DenseVector) LinAlgebraIOUtils.loadVectorFromObject(pathPrefix + "_mean", sc);
		
		// second file name is the ZCA
		ZCA = (DenseMatrix) LinAlgebraIOUtils.loadMatrixFromObject(pathPrefix + "_zca", sc);

	}
	
	
	/**
	 * Saves the fields necessary to reconstruct a preprocessor object. 
	 * First file name is the mean vector, second file name is the ZCA matrix.
	 * 
	 * @param pathPrefix Path prefix to save the pre-processing variables.
	 * @param sc Spark context variable.
	 **/
	@Override
	public void saveToFile(String pathPrefix, JavaSparkContext sc) {

		// first file name is the mean
		LinAlgebraIOUtils.saveVectorToObject(this.mean, pathPrefix + "_mean", sc);
		
		// second file name is the ZCA matrix
		LinAlgebraIOUtils.saveMatrixToObject(this.ZCA, pathPrefix + "_zca", sc);
	}
}


/**
 * Helper class that does the projection of the original data onto the whitened space. 
 * It uses a map that takes as input a tuple of vectors and projects the second part 
 * of the pair to the whitened space. It outputs a tuple of vectors where the 
 * first part of the pair remains the same.
 */
class convert2Tuple2 implements Function<Tuple2<Vector, Vector>, Tuple2<Vector, Vector>> {

	private static final long serialVersionUID = 3141810027556534234L;

	private DenseVector m;
	private DenseMatrix M;
	

	/**
	 * Constructor that sets the mean vector and the ZCA matrix.
	 * 
	 * @param M ZCA matrix.
	 * @param m Mean vector.
	 */
	public convert2Tuple2(DenseVector m, DenseMatrix M) {
		this.m = m;
		this.M = M;
	}

	
	/**
	 * Method that project the second part of the data tuples onto the new space. 
	 * It assembles together the first tuple together with the projected data 
	 * and returns a new tuple.
	 * 
	 * @param pair Input tuple.
	 * @return New tuple with updated projected data.
	 */
	public Tuple2<Vector, Vector> call(Tuple2<Vector, Vector> pair) {

		// compute multiplication between the second part of the pair and the DenseMatrix
		DenseVector v = (DenseVector) pair._2;

		// subtract the mean
		v = MatrixOps.localVecSubtractMean(v, m);

		// project to the whitened space
		DenseVector out = new DenseVector(new double[v.size()]);
		BLAS.gemv(1.0, M.transpose(), v, 0, out);

		return new Tuple2<Vector, Vector>(pair._1, out);
	}
}
