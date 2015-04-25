package test.java;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import junit.framework.Assert;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
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
	public void testDenseVectorIO() {
		// Create a sample mean vector 
		Vector mean = Vectors.dense(1,2,3,4);
		List<Vector> temp_mean = new ArrayList<Vector>();
		temp_mean.add(mean);
		String filename = "tmp3";
		// Transform it to JavaRDD and save it to tmp1_mean file
		sc.parallelize(temp_mean).saveAsTextFile(filename + "_mean");
		
		// Read back the file as an array of strings
		JavaRDD<String> mean_read = sc.textFile(filename + "_mean");
		
		List<String> mean_string = mean_read.collect();
		assert(mean_string.size() == 1);
		System.out.println(mean_string);
		String m = mean_string.get(0);
		m = m.substring(1, m.length()-2);
		String[] parts = m.split(",");
		int vector_size = parts.length;
		double mean_vector[] = new double[vector_size];
		for (int i = 0; i < vector_size; ++i) {
			mean_vector[i] = Double.parseDouble(parts[i]);
		}
		Vector reconstructed = Vectors.dense(mean_vector);
		Assert.assertEquals(mean.toString(), reconstructed.toString());
	}

}
