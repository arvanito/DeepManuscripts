package test.java;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import junit.framework.Assert;
import main.java.LinAlgebraIOUtils;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class LoadSaveModelTest {
	private transient JavaSparkContext sc;
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "LoadSaveTest");
	}

	/**
	 * @throws java.lang.Exception
	 */
	@After
	public void tearDown() throws Exception {
		sc.stop();
		sc = null;
	}

	@Test @Ignore
	public void testDenseVectorTextIO() {
		String filename = "temp3";
		// Create a sample mean vector 
		DenseVector mean = (DenseVector) Vectors.dense(1,2,3,4);
		LinAlgebraIOUtils.saveVectorToText(mean, filename, sc);
		
		// Read back the file as an array of strings
		Vector reconstructed = LinAlgebraIOUtils.loadVectorFromText(filename, sc);
		Assert.assertEquals(mean.toString(), reconstructed.toString());
	}
	@Test
	public void testDenseVectorObjectIO() {
		String filename = "temp2";
		// Create a sample mean vector 
		DenseVector input = (DenseVector) Vectors.dense(1,2,3,4);
		LinAlgebraIOUtils.saveVectorToObject(input, filename, sc);
		
	    // Read back the array
		Vector reconstructed = LinAlgebraIOUtils.loadVectorFromObject(filename, sc);
		Assert.assertEquals(input.toString(), reconstructed.toString());
	}
	@Test @Ignore
	public void testDenseMatrixObjectIO() {
		String filename = "tmp6";
		// Create a sample mean vector 
		Matrix input = Matrices.dense(2, 3, new double[] {1,2,3,4,5,6});
		LinAlgebraIOUtils.saveMatrixToObject(input, filename, sc);
		
	    // Read back the array
		Matrix reconstructed = LinAlgebraIOUtils.loadMatrixFromObject(filename, sc);
		Assert.assertEquals(input.toString(), reconstructed.toString());
		Assert.assertEquals(2, reconstructed.numRows());
		Assert.assertEquals(3, reconstructed.numCols());
		Assert.assertEquals(6.0, reconstructed.apply(1, 2));
	}
	
	@Ignore @Test
	public void testDenseMatrixTextIO() {
		// Test is not currently working
		String filename = "tmp7";
		// Create a sample mean vector 
		Matrix input = Matrices.dense(2, 3, new double[] {1,2,3,4,5,6});
		LinAlgebraIOUtils.saveMatrixToText(input, filename, sc);
		
	    // Read back the array
		Matrix reconstructed = LinAlgebraIOUtils.loadMatrixFromText(filename, sc);
		Assert.assertEquals(input.toString(), reconstructed.toString());
		Assert.assertEquals(2, reconstructed.numRows());
		Assert.assertEquals(3, reconstructed.numCols());
		Assert.assertEquals(6.0, reconstructed.apply(1, 2));
	}
}
