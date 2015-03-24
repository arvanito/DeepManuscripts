package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;

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

/**
 * Class that performs ZCA whitening on the dataset.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class PreProcessZCA implements PreProcessor {

	private ConfigBaseLayer configLayer;
	private DenseVector mean;
	private DenseMatrix ZCA;
	
	
	/**
	 * Getter method for the ConfigBaseLayer object.
	 * 
	 * @return The ConfigBaseLayer object
	 */
	public ConfigBaseLayer getConfigLayer() {
		return configLayer;
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
	 * Setter for the mean vector.
	 * 
	 * @param mean Mean vector
	 */
	public void setMean(DenseVector mean) {
		this.mean = mean;
	}
	
	
	/**
	 * Setter for the ZCA matrix.
	 * 
	 * @param ZCA The ZCA matrix
	 */
	public void setZCA(DenseMatrix ZCA) {
		this.ZCA = ZCA;
	}
	
	
	/**
	 * Main method that preprocesses the dataset. 
	 * 
	 * @param data Input distributed dataset
	 * @param configLayer Current layer configuration from the protocol buffer
	 * @return Preprocessed distributed dataset
	 */
	@Override
	public JavaRDD<Vector> preprocessData(JavaRDD<Vector> data,
			ConfigBaseLayer configLayer) {

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
		
		// perform ZCA whitening and project the data onto the decorrelated space
		DenseMatrix ZCA = performZCA(rowData, configLayer.getConfigPreprocess().getEps2());
		rowData = rowData.multiply(ZCA);
		setZCA(ZCA);
			
		// convert the distributed RowMatrix into a JavaRDD<Vector> 
		data = new JavaRDD<Vector>(rowData.rows(), data.classTag());

		return data;
	}

	
	/**
	 * Method that performs ZCA whitening on the data.
	 * 
	 * @param mat Input distributed RowMatrix
	 * @param e Parameter for eigenvalue normalization
	 * @return ZCA matrix of type DenseMatrix
	 */
	public DenseMatrix performZCA(RowMatrix mat, double e) {
		
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
		BLAS.gemm(false, true, 1.0, Matrices.diag(Vectors.dense(l)), V, 0.0, ZCA);
		BLAS.gemm(false, false, 1.0, V, ZCA, 0.0, ZCA);
		
		return ZCA;
	}
	
}
