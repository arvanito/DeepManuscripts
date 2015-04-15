package test.java;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.Serializable;

import main.java.FFT;
import main.java.FFTConvolutionExtractor;
import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigFeatureExtractor;
import main.java.DeepModelSettings.ConfigPooler;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class FFTConvolutionTest implements Serializable {
	
	private static final long serialVersionUID = 2780960378816038954L;
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

	public boolean deepApproximateEquals(double[][][] x, double[][][] y, double epsilon) {
		if(x.length != y.length) fail("1D lengths differ");
		for(int i = 0; i < x.length; ++i) {
			if(x[i].length != y[i].length) fail("2D lengths differ at " + i);
			for(int j = 0; j < x[i].length; ++j) {
				if(x[i][j].length != y[i][j].length) fail("3D lengths differ at " + i + " , " + j);
				for(int k = 0; k < x[i][j].length; ++k) {
					if(Math.abs(x[i][j][k] - y[i][j][k]) > epsilon ) {
						fail("Values at " + i + "-" + j + "-" + k + " differ: " +  x[i][j][k] + " vs. " + y[i][j][k]);
						return false;
					}
				}
			}
		}
		return true;
	}
	
	@Test
	public void simpleTest1DFFT() {
		double[][] input =   {{1,2,3,0}};
		double[][][] output = FFT.fftReal1Dor2D(input);
		double[][][] expected = {{{6,0},{-2,-2},{2,0},{-2,2}}};
		//System.out.println(Arrays.deepToString(output));
		assertTrue(deepApproximateEquals(output, expected, 1e-2));
	}
	
	@Test
	public void simpleTest2DFFT() {
		double[][] input =   {{1,2,3,0}, {4,5,6,0}, {7,8,9,0},{0,0,0,0}};
		double[][][] output = FFT.fftReal1Dor2D(input);
		//System.out.println(Arrays.deepToString(output));
		double[][][] expected = {{{ 45,  0}, { -6,-15}, { 15,  0}, { -6, 15}},
				 {{-18,-15}, { -5,  8}, { -6, -5}, {  5, -4}},
				 {{ 15,  0}, { -2, -5}, {  5,  0}, { -2,  5}},
				 {{-18, 15}, {  5,  4}, { -6,  5}, { -5, -8}}};

		assertTrue(deepApproximateEquals(output, expected, 1e-2));
	}
	
	@Test
	public void simpleTest1DConvolution() {
		int inputRows = 1;
		int inputCols = 4;
		int featureRows = 1;
		int featureCols = 2;
		double[] input =   {1,2,3,0};
	    
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
				setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(featureCols).setFeatureDim2(featureRows).
						                  setInputDim1(inputCols).setInputDim2(inputRows)).
			    setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).build();
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf, null); // no pre-processing yet

		Vector data = new DenseVector(input);
		
		double[] A = {1,2};
		
		Vector[] features = {new DenseVector(A)};
		extractor.setFeatures(features);
		Vector output;
		try {
			output = extractor.call(data);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Call threw exception");
			return;
		}
		//System.out.println(Arrays.toString(output.toArray()));
		double[] expected_outputs = {5,8,3};
		Assert.assertArrayEquals(expected_outputs, output.toArray(), 1e-6);
	}
	
	@Test
	public void simpleTest2DConvolution() {
		int inputRows = 4;
		int inputCols = 4;
		int featureRows = 2;
		int featureCols = 2;
		double[] input =   {1,2,3,0,
							4,5,6,0,
							7,8,9,0,
							0,0,0,0};
	    
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
				setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(featureCols).setFeatureDim2(featureRows).
						                  setInputDim1(inputCols).setInputDim2(inputRows)).
			    setConfigPooler(ConfigPooler.newBuilder().setPoolSize(1)).build();
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf, null); // no pre-processing yet

		Vector data = new DenseVector(input);
		
		double[] A = {1,2,
					  0,0};
		
		Vector[] features = {new DenseVector(A)};
		extractor.setFeatures(features);
		Vector output;
		try {
			output = extractor.call(data);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Call threw exception");
			return;
		}
		//System.out.println(Arrays.toString(output.toArray()));
		double[] expected_outputs = {5,8,3,
									 14,17,6,
									 23,26,9};
		Assert.assertArrayEquals(expected_outputs, output.toArray(), 1e-6);
	}
	
	// TODO Write test case that uses pre-processing
	
	/*// Meh boooring
	@Test
	public void test2DConvolution() {
		int inputRows = 8;
		int inputCols = 16;
		int featureRows = 4;
		int featureCols = 4;
		double[] input =  {0.7803,   0.5752,   0.6491,   0.6868,   0.4868,   0.6443,   0.6225,   0.2259,   0.9049,   0.6028,   0.0855,   0.2373,   0.6791,   0.0987,   0.4942,   0.0305,
	    					0.3897,   0.0598,   0.7317,   0.1835,   0.4359,   0.3786,   0.5870,   0.1707,   0.9797,   0.7112,   0.2625,   0.4588,   0.3955,   0.2619,   0.7791,   0.7441,
	    					0.2417,   0.2348,   0.6477,   0.3685,   0.4468,   0.8116,   0.2077,   0.2277,   0.4389,   0.2217,   0.8010,   0.9631,   0.3674,   0.3354,   0.7150,   0.5000,
	    					0.4039,   0.3532,   0.4509,   0.6256,   0.3063,   0.5328,   0.3012,   0.4357,   0.1111,   0.1174,   0.0292,   0.5468,   0.9880,   0.6797,   0.9037,   0.4799,
	    					0.0965,   0.8212,   0.5470,   0.7802,   0.5085,   0.3507,   0.4709,   0.3111,   0.2581,   0.2967,   0.9289,   0.5211,   0.0377,   0.1366,   0.8909,   0.9047,
	    					0.1320,   0.0154,   0.2963,   0.0811,   0.5108,   0.9390,   0.2305,   0.9234,   0.4087,   0.3188,   0.7303,   0.2316,   0.8852,   0.7212,   0.3342,   0.6099,
	    					0.9421,   0.0430,   0.7447,   0.9294,   0.8176,   0.8759,   0.8443,   0.4302,   0.5949,   0.4242,   0.4886,   0.4889,   0.9133,   0.1068,   0.6987,   0.6177,
	    					0.9561,   0.1690,   0.1890,   0.7757,   0.7948,   0.5502,   0.1948,   0.1848,   0.2622,   0.5079,   0.5785,   0.6241,   0.7962,   0.6538,   0.1978,   0.8594};
	    
		ConfigBaseLayer conf = ConfigBaseLayer.newBuilder().
				setConfigFeatureExtractor(ConfigFeatureExtractor.newBuilder().
						                  setFeatureDim1(featureCols).setFeatureDim2(featureRows).
						                  setInputDim1(inputCols).setInputDim2(inputRows)).build();
		FFTConvolutionExtractor extractor = new FFTConvolutionExtractor(conf, null); // no pre-processing yet

		Vector data = new DenseVector(input);
		
		double[] A = {	0.8055,    0.8865,    0.9787,    0.0596,
			    		0.5767,    0.0287,    0.7127,    0.6820,
			    		0.1829,    0.4899,    0.5005,    0.0424,
			    		0.2399,    0.1679,    0.4711,    0.0714};
		
		Vector[] features = {new DenseVector(A)};
		
		Vector output;
		try {
			output = extractor.call(data);
		} catch (Exception e) {
			e.printStackTrace();
			fail("Call threw exception");
		}
		double[] expected_outputs = {8,10,12,20,22,24};
		Assert.assertArrayEquals(expected_outputs, output.toArray(), 1e-6);
	}
*/

}
