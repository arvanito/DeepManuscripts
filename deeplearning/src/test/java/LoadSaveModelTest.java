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

	@Ignore @Test
	public void testDenseVectorTextIO() {
		// Create a sample mean vector 
		Vector mean = Vectors.dense(1,2,3,4);
		String filename = "tmp6";
		LinAlgebraIOUtils.saveToText(mean, filename + "_mean", sc);
		
		// Read back the file as an array of strings
		Vector reconstructed = LinAlgebraIOUtils.loadFromText(filename + "_mean", sc);
		Assert.assertEquals(mean.toString(), reconstructed.toString());
	}
	@Ignore @Test
	public void testDenseVectorObjectIO() {
		// Create a sample mean vector 
		Vector mean = Vectors.dense(1,2,3,4);
		List<Vector> temp_mean = new ArrayList<Vector>();
		temp_mean.add(mean);
		sc.parallelize(temp_mean).saveAsObjectFile("t2");
		
	    // Read back the array
		JavaRDD<Vector> out = sc.objectFile("t2");
		List<Vector> a = out.toArray();
		Assert.assertEquals(mean.toString(),a.get(0).toString());
	}

}
