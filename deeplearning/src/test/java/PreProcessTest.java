package test.java;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import main.java.ContrastNormalization;
import main.java.MatrixOps;
import main.java.PreProcessZCA;
import main.java.PreProcessor;
import main.java.SubtractMean;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
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
		//double[] data = {0.76, 0.34, 0.12, 0.32};
		//double[] x1 = {0.56, 0.54, 1.23, 0.57};
		//double[] x2 = {0.34, 0.63, 0.34, 0.85};
		//double[] x3 = {0.32, 1.2, 0.67, 0.29};
		//double[] x4 = {0.14, 0.78, 0.85, 0.94};
		double[] x1 = {0.56, 0.34, 0.32, 0.14};
		double[] x2 = {0.54, 0.63, 1.2, 0.78};
		double[] x3 = {1.23, 0.34, 0.67, 0.85};
		double[] x4 = {0.57, 0.85, 0.29, 0.94};
		double[] x = {0.56, 0.54, 1.23, 0.57, 0.34, 0.63, 0.34, 0.85, 0.32, 1.2, 0.67, 0.29, 0.14, 0.78, 0.85, 0.94};
		//double[] m = {0.32, 0.53, 0.12, 0.13};
		//double[] expected_output = {-1.114153311705324, -0.705279543151641, -1.170587714869567, -1.330205911918481};
		//double[] expected_output = {1.6546,    0.5420,    0.5200,    0.4457,
		//	    					0.5420,    1.9192,    0.5226,    0.1785,
		//	    					0.5200,    0.5226,    1.6011,    0.5186,
		//	    					0.4457,    0.1785,    0.5186,    2.0195};
		
		
		double[] expected_output = {1.654633794518243,   0.541992148747697,   0.519961336130961,   0.445690380771477,
									0.541992148747697,   1.919200272146810,   0.522623043139470,   0.178462196134402,
									0.519961336130961,   0.522623043139470,   1.601056255503961,   0.518637025393985,
									0.445690380771477,   0.178462196134402,   0.518637025393985,   2.019488057868512};
		
		// create a parallel dataset from the local matrix
		List<Vector> matX = new ArrayList<Vector>(4);
		matX.add(Vectors.dense(x1));
		matX.add(Vectors.dense(x2));
		matX.add(Vectors.dense(x3));
		matX.add(Vectors.dense(x4));
		JavaRDD<Vector> matRDD = sc.parallelize(matX);
		
		// do contrast normalization
		matRDD = matRDD.map(new ContrastNormalization(0.1));
		
		// create a RowMatrix from local data
		RowMatrix mat = new RowMatrix(matRDD.rdd());
		
		// compute mean data Vector
		MultivariateStatisticalSummary summary = mat.computeColumnSummaryStatistics();
		DenseVector m = (DenseVector) summary.mean();

		// remove the mean from the dataset
		matRDD = matRDD.map(new SubtractMean(m));
		
		// make dense vectors in spark format
		//DenseVector dataDense = new DenseVector(data);
		//DenseVector dataOut = new DenseVector(new double[data.length]);
		//DenseMatrix zca = new DenseMatrix(4,4,x);
		//DenseVector mean = new DenseVector(m);
		
		// create a RowMatrix from local data
		mat = new RowMatrix(matRDD.rdd());
		
		// create a PreProcessZCA object and call the performZCA
		PreProcessZCA preprocess = new PreProcessZCA();
		DenseMatrix zca = preprocess.performZCA(mat, 0.1);
		
				
		// do the pre-processing
		//dataDense = MatrixOps.localVecContrastNorm(dataDense, 0.1);
		//dataDense = MatrixOps.localVecSubtractMean(dataDense, mean);
		//BLAS.gemv(1.0, zca.transpose(), dataDense, 0.0, dataOut);
		//System.out.println("Data out:");
		//System.out.println(dataOut);
		
		//DenseVector zcaDense = zca.transpose().multiply(dataDense);
		//System.out.println("ZCA dense:");
		//System.out.println(zcaDense);
		
		Assert.assertArrayEquals(expected_output, zca.toArray(), 1e-6);
		
	}
	

}
