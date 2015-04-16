package test.java;

import java.io.Serializable;

import main.java.MatrixOps;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class PreProcessTest implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -7766242513300032083L;
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
	public void preprocessTest() {
		
		// simple test example for pre-processing
		double[] data = {0.76, 0.34, 0.12, 0.32};
		double[] x = {0.56, 0.54, 1.23, 0.57, 0.34, 0.63, 0.34, 0.85, 0.32, 1.2, 0.67, 0.29, 0.14, 0.78, 0.85, 0.94};
		double[] m = {0.32, 0.53, 0.12, 0.13};
		double[] expected_output = {-1.1142, -0.7053, -1.1706, -1.3302};
		
		// make dense vectors in spark format
		DenseVector dataDense = new DenseVector(data);
		DenseVector dataOut = new DenseVector(new double[data.length]);
		DenseMatrix zca = new DenseMatrix(4,4,x);
		DenseVector mean = new DenseVector(m);
		
		// do the pre-processing
		dataDense = MatrixOps.localVecContrastNorm(dataDense, 0.1);
		dataDense = MatrixOps.localVecSubtractMean(dataDense, mean);
		
		BLAS.gemv(1.0, zca.transpose(), dataDense, 0.0, dataOut);
		System.out.println("Data out:");
		System.out.println(dataOut);
		
		//DenseVector zcaDense = zca.transpose().multiply(dataDense);
		//System.out.println("ZCA dense:");
		//System.out.println(zcaDense);
		
		Assert.assertArrayEquals(expected_output, dataOut.toArray(), 1e-6);
		
	}
	

}
