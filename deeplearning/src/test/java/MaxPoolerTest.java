/**
 * 
 */
package test.java;

import static org.junit.Assert.*;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import main.java.DeepLearningLayer;
import main.java.DeepModelSettings.ConfigPooler;
import main.java.DummyLayer;
import main.java.MaxPooler;
import main.java.Parse;
import main.java.DeepModelSettings.ConfigBaseLayer;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * @author Viviana Petrescu
 *
 */
public class MaxPoolerTest implements Serializable{
	private static final long serialVersionUID = 145346357547456L;
	private transient JavaSparkContext sc;
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		sc = new JavaSparkContext("local", "MaxPoolerTest");
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
	public void sampleTest() {
		//fail("Not yet implemented");
		List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
		JavaRDD<Integer> distData = sc.parallelize(data);
		class Sum implements Function2<Integer, Integer, Integer>, Serializable {
			private static final long serialVersionUID = 2685928850298905497L;

			public Integer call(Integer a, Integer b) { return a + b; }
		}

		int totalLength = distData.reduce(new Sum());
		Assert.assertEquals(15, totalLength);
	}
	@Test
	public void test1DMaxPooling() {
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().setConfigPooler(ConfigPooler.newBuilder().setPoolSize(2)).build();
		MaxPooler pooler = new MaxPooler(conf);
		double[] input = {1,2,3,4,5,6,7,8,9,10};
		Vector data = Vectors.dense(input);
		Vector output = pooler.poolOver1D(data);
		Assert.assertEquals(5, output.size());
		double[] expected_outputs = {2,4,6,8,10};
		Assert.assertArrayEquals(expected_outputs, output.toArray(), 1e-6);
	}

}
