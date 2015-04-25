package test.java;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import junit.framework.Assert;
import main.java.LinAlgebraIOUtils;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
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

	@Test
	public void testDenseVectorTextIO() {
		String filename = "temp3";
		// Create a sample mean vector 
		Vector mean = Vectors.dense(1,2,3,4);
		LinAlgebraIOUtils.saveToText(mean, filename, sc);
		
		// Read back the file as an array of strings
		Vector reconstructed = LinAlgebraIOUtils.loadFromText(filename, sc);
		Assert.assertEquals(mean.toString(), reconstructed.toString());
	}
	@Test
	public void testDenseVectorObjectIO() {
		String filename = "temp2";
		// Create a sample mean vector 
		Vector input = Vectors.dense(1,2,3,4);
		LinAlgebraIOUtils.saveToObject(input, filename, sc);
		
	    // Read back the array
		Vector reconstructed = LinAlgebraIOUtils.loadFromObject(filename, sc);
		Assert.assertEquals(input.toString(), reconstructed.toString());
	}

}
