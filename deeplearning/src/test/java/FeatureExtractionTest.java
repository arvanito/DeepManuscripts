package test.java;

import java.io.Serializable;
import java.util.Arrays;

import main.java.MatrixOps;
import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.PreProcessZCA;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class FeatureExtractionTest implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5361837911584977475L;
	private transient JavaSparkContext sc;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "FeatureExtractionTest");
	}
	
	/**
	 * @throws java.lang.Exception
	 */
	@After
	public void tearDown() throws Exception {
		sc.stop();
		sc = null;
	}
	
	@Test
	public void multiplyTest() {
		//ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().setConfigFeatureExtractor()
		
		// simple example
		double[] x = {0.1, 0.2, 0.4, 1.4, 2.3};
		double[] f1 = {0.56, 0.34, 0.32, 0.14, 0.25};
		double[] f2 = {0.54, 0.63, 1.2, 0.78, 1.23};
		double[] f3 = {1.23, 0.34, 0.67, 0.85, 0.43};
		double[] expected_output = {1.0230, 4.5810, 2.6380};
		
		// create Vectors from double arrays
		Vector vx = Vectors.dense(x);
		DenseVector dvx = (DenseVector) vx;
		
 		Vector[] vf = new Vector[3];
		vf[0] = Vectors.dense(f1);
		vf[1] = Vectors.dense(f2);
		vf[2] = Vectors.dense(f3);
	
		// run the feature extraction code
		DenseMatrix D = MatrixOps.convertVectors2Mat(vf);
		DenseVector dvxOut = new DenseVector(new double[D.numRows()]);
		BLAS.gemv(1.0, D, dvx, 0.0, dvxOut);
		
		Assert.assertArrayEquals(expected_output, dvxOut.toArray(), 1e-6);
		
	}
	
	
	@Test
	public void multiplyPreTest() {
		
		// simple example
		double[] x1 = {0.56, 0.34, 0.32, 0.14};
		double[] x2 = {0.54, 0.63, 1.2, 0.78};
		double[] x3 = {1.23, 0.34, 0.67, 0.85};
		double[] x4 = {0.57, 0.85, 0.29, 0.94};
		double[] x = {0.56, 0.54, 1.23, 0.57, 0.34, 0.63, 0.34, 0.85, 0.32, 1.2, 0.67, 0.29, 0.14, 0.78, 0.85, 0.94};
		
		double[] zca = {1.654633794518243,   0.541992148747697,   0.519961336130961,   0.445690380771477,
				0.541992148747697,   1.919200272146810,   0.522623043139470,   0.178462196134402,
				0.519961336130961,   0.522623043139470,   1.601056255503961,   0.518637025393985,
				0.445690380771477,   0.178462196134402,   0.518637025393985,   2.019488057868512};
		double[] m = {0.190168010867027,  -0.204704063143829,  -0.042614218658781,   0.057150270935583};
		
		DenseVector mean = new DenseVector(m);
		DenseMatrix ZCA = new DenseMatrix(4,4,zca);
		
		// create a PreProcessZCA object with the input mean and ZCA variables
		PreProcessZCA preProcess = new PreProcessZCA(mean, ZCA);
		
		
	}
	
	
	@Test
	public void convMultiplyTest() {
		//ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().setConfigFeatureExtractor()
		//ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
		//		setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
		//				                  setFeatureDim1(2).setFeatureDim2(2).setInputDim1(4).setInputDim2(4)).
		//				                  setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2)).build();
		
		// simple example
		double[] f1 = {0.1, 0.2, 0.4, 1.4};
		double[] f2 = {0.5, 0.2, 0.1, 0.5};
		double[] x = {0.56, 0.54, 1.23, 0.57, 0.34, 0.63, 0.34, 0.85, 0.32, 1.2, 0.67, 0.29, 0.14, 0.78, 0.85, 0.94};
		//double[] expected_output = {1.1310, 1.4080, 2.1030, 0.9120, 1.3250, 0.9790, 1.0340, 2.1890, 1.3180};
	    double[] expected_output = {1.1820, 1.0280, 1.5630, 1.9680, 1.5490, 0.8780, 1.4200, 1.7560, 1.7810};
		
	    // create Vectors from double arrays
		Vector vx = Vectors.dense(x);
		
 		Vector[] vf = new Vector[1];
		vf[0] = Vectors.dense(f1);
		//vf[1] = Vectors.dense(f2);
	
		// run the feature extraction code
		DenseMatrix D = MatrixOps.convertVectors2Mat(vf);
		int[] dims = {4,4};
		int[] rfSize = {2,2};
		DenseMatrix M = MatrixOps.reshapeVec2Mat((DenseVector) vx, dims);	
		DenseMatrix patches = MatrixOps.im2colT(M, rfSize);
		
		// allocate memory for the output vector
		DenseMatrix out = new DenseMatrix(patches.numRows(),D.numRows(),new double[patches.numRows()*D.numRows()]);	
		// multiply the matrix of the learned features with the preprocessed data point
		BLAS.gemm(1.0, patches, D.transpose(), 0.0, out);
		//DenseVector outVec = MatrixOps.reshapeMat2Vec(out);
		
		Assert.assertArrayEquals(expected_output, out.toArray(), 1e-6);
		
	}
	
}
