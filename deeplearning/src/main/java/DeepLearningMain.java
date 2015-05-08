package main.java;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

import com.google.protobuf.TextFormat;

import main.java.DeepModelSettings.*;

/**
 * Main class.
 * 
 */
public class DeepLearningMain {
	
	/**
	 * Method that loads the layer configurations from .prototxt file. 
	 * Makes use of the protocol buffers for describing the configuration. 
	 * 
	 * @param prototxt_file Input protocol buffer configuration
	 * @return List of objects that describe the configuration in each layer
	 */
	public static List<ConfigBaseLayer> loadSettings(String prototxt_file) {
		
		List<ConfigBaseLayer> globalConfig = null;
		try {
			ConfigManuscripts.Builder builder = ConfigManuscripts.newBuilder();
			//BufferedReader reader = new BufferedReader(new FileReader(prototxt_file));
			FileInputStream fs = new FileInputStream(prototxt_file);
			InputStreamReader reader = new InputStreamReader(fs);
			TextFormat.merge(reader, builder);
			
			// Settings file created
			ConfigManuscripts settings = builder.build();
			System.out.printf("# base layers is %d\n", settings.getConfigLayerCount());
			
			globalConfig = settings.getConfigLayerList();
			for (ConfigBaseLayer blayer: globalConfig) {
				if (blayer.hasConfigKmeans()) {
					System.out.printf("Kmeans clusters %d\n", blayer.getConfigKmeans().getNumberOfClusters());
				}
				if (blayer.hasConfigAutoencoders()) {
					System.out.printf("Autoencoders units %d\n", blayer.getConfigAutoencoders().getNumberOfUnits());
				}
				ConfigPooler cpooler = blayer.getConfigPooler();
				System.out.printf("Pool size %d\n", cpooler.getPoolSize());
			}
			
			reader.close();

		} catch (IOException e) {
			System.err.println("Input .prototxt parsing failed");
			System.err.println(prototxt_file);
			e.printStackTrace();
			System.exit(1);
		}

		return globalConfig;
	}
	
	/**
	 * Main method that trains the model layer by layer. 
	 * 
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration
	 * @param inputFileSmallPatches Small dataset input, represents small patch learning for the first layer
	 * @param inputFileLargePatches Large dataset input, represents high-level learning for the remaining layers
	 * @param outputFile File to save the final pooled representations after full training
	 * @throws Exception
	 */
	public static void train(List<ConfigBaseLayer> globalConfig, String inputFileSmallPatches, 
							String inputFileLargePatches, String outputFile) throws Exception {
		
		// open files and convert them to JavaRDD<Vector> datasets
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
 
		JavaRDD<Tuple2<Vector, Vector>> inputSmallPatches = sc.textFile(inputFileSmallPatches).map(new ParseTuples());
		JavaRDD<Tuple2<Vector, Vector>> inputWordPatches = sc.textFile(inputFileLargePatches).map(new ParseTuples());

		// The main loop calls train() on each of the layers
		JavaRDD<Tuple2<Vector, Vector>> result = null;
	 	for (int layerIndex = 0; layerIndex < globalConfig.size(); ++layerIndex) {
	 		
	 		// set up the current layer 
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(globalConfig.get(layerIndex), layerIndex, "x");
			
			// The configLayer has configExtractor only if it convolutional,
			// The multiply Extractor does not need any parameters.
			if (globalConfig.get(layerIndex).hasConfigFeatureExtractor()) {
				result = layer.train(inputSmallPatches, inputWordPatches);
			} else {
				result = layer.train(result, result);
			}	
	 	}
		//TODO save also last file
		result.saveAsTextFile(outputFile);
		
		sc.close();
	}
	
	
	/**
	 * Main method for testing the trained model. 
	 * 
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration
	 * @param inputFile Input file that contains the test patches for comparison
	 * @param outputFile Output file that contains the final representations of the test patches
	 */
	public static void test(List<ConfigBaseLayer> globalConfig, String inputFile, String outputFile) throws Exception {
		
		// open the test file and convert it to a JavaRDD<Vector> dataset
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
    	
		JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.textFile(inputFile).map(new ParseTuples());
    	
		// The main loop calls test() on each of the layers
		JavaRDD<Tuple2<Vector, Vector>> result = null;
	 	for (int layerIndex = 0; layerIndex < globalConfig.size(); ++layerIndex) {
	 		
	 		// set up the current layer 
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(globalConfig.get(layerIndex), layerIndex, "x");
			
			// The configLayer has configExtractor only if it convolutional,
			// The multiply Extractor does not need any parameters.
			if (globalConfig.get(layerIndex).hasConfigFeatureExtractor()) {
				result = layer.test(testPatches);
			} else {
				result = layer.test(result);
			}	
	 	}
	 	
	 	//TODO save the result to a file 
	 	
		sc.close();
	}
	
	
	/**
	 * Main method for testing the trained model. 
	 * 
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration
	 * @param testPatches JavaRDD<Tuple2<Vector, Vector>> which contains all the candidate patches
	 * @return The resulting representations of the test patches
	 */
	public static JavaRDD<Tuple2<Vector, Vector>> testRDD(List<ConfigBaseLayer> globalConfig, JavaRDD<Tuple2<Vector, Vector>> testPatches) throws Exception {
		
		// The main loop calls test() on each of the layers
		JavaRDD<Tuple2<Vector, Vector>> result = null;
	 	for (int layerIndex = 0; layerIndex < globalConfig.size(); ++layerIndex) {
	 		
	 		// set up the current layer 
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(globalConfig.get(layerIndex), layerIndex, "x");
			
			// The configLayer has configExtractor only if it convolutional,
			// The multiply Extractor does not need any parameters.
			if (globalConfig.get(layerIndex).hasConfigFeatureExtractor()) {
				result = layer.test(testPatches);
			} else {
				result = layer.test(result);
			}	
	 	}
	 	
	 	// return the final representations of the test patches
	 	return result;
	}
	
	
	/**
	 * Main method for ranking the candidate patches with the query. 
	 * 
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration
	 * @param inputFileQuery Input file that contains the query
	 * @param inputFilePatches Input file that contains the candidate patches
	 */
	public static void rank(List<ConfigBaseLayer> globalConfig, String inputFileQuery, String inputFilePatches) throws Exception {
		
		// open the test file and convert it to a JavaRDD<Vector> dataset
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
    	
		// load query 
		JavaRDD<Tuple2<Vector, Vector>> query = sc.textFile(inputFileQuery).map(new ParseTuples());
		Tuple2<Vector, Vector> queryTuple = query.collect().get(0);
		//JavaRDD<Tuple2<Vector, Vector>> query = sc.objectFile(inputFileQuery);
    	
		// load candidate patches
		JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.textFile(inputFilePatches).map(new ParseTuples());
		//JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.objectFile(inputFilePatches);
		
		// get the dimensions of the large learned patches
		int[] inputDims = {globalConfig.get(0).getConfigFeatureExtractor().getInputDim1(), globalConfig.get(0).getConfigFeatureExtractor().getInputDim2()};
		
		// extract (overlapping?) patches from the input query and candidate patches
		Vector vecSize = queryTuple._1;
		int[] vecSizeInt = new int[vecSize.size()];
		for (int i = 0; i < vecSize.size(); i++) {
			vecSizeInt[i] = (int) vecSize.apply(i);
		}
		JavaRDD<Tuple2<Vector, Vector>> queryPatches = query.flatMap(new ExtractPatches(vecSizeInt, inputDims));
		testPatches = testPatches.flatMap(new ExtractPatches(vecSizeInt, inputDims));
		
		// compute representations for the query and candidate patches
		JavaRDD<Tuple2<Vector, Vector>> queryReps = testRDD(globalConfig, queryPatches);
		JavaRDD<Tuple2<Vector, Vector>> testReps = testRDD(globalConfig, testPatches);
		
		// concatenate patch representations to represent the original query
		List<Tuple2<Vector, Vector>> queryRepList = queryReps.collect();
		int queryRepLength = queryRepList.size();			// size of the list
		int vecLength = queryRepList.get(0)._2.size();		// size of the patch representations in the list
		double[] queryData = new double[queryRepLength*vecLength];

		int k = 0;	// offset for each Vector in the List
		for (int i = 0; i < queryRepLength; i++) {
			System.arraycopy(queryRepList.get(i)._2, 0, queryData, k, vecLength);
			k = k + vecLength;
		}
		Tuple2<Vector, Vector> queryRep = new Tuple2<Vector, Vector>(queryTuple._1, Vectors.dense(queryData));
		
		// concatenate patch representations from the RDD
		JavaPairRDD<Vector, Vector> testRepsPair = JavaPairRDD.fromJavaRDD(testReps);
		testRepsPair = testRepsPair.reduceByKey(
				new Function2<Vector, Vector, Vector>() {
					private static final long serialVersionUID = 5851620097719920872L;

					// concatenate two Vectors with the same key
					public Vector call(Vector v1, Vector v2) {
						double[] out = new double[v1.size() + v2.size()];

						System.arraycopy(v1.toArray(), 0, out, 0, v1.size());
						System.arraycopy(v2.toArray(), 0, out, v1.size(), v2.size());

						return Vectors.dense(out);
					}
				}
			);
		testReps = testRepsPair.rdd().toJavaRDD();	// convert back to JavaRDD<Tuple2<Vector, Vector>>
		
		// compute cosine similarities between the query representation and the candidate patches' representations
		JavaRDD<Tuple2<Vector, Double>> cosineSim = testReps.map(new ComputeSimilarityPair(queryRep));
		
		// rank the results
		List<Tuple2<Vector, Double>> simList = cosineSim.collect();
		int simListSize = simList.size();
		
		// extract in separate Lists the Vector metadata and Double similarities
		final List<Vector> simMeta = new ArrayList<Vector>(simListSize);
		final List<Double> simData = new ArrayList<Double>(simListSize);
		for (int i = 0; i < simListSize; i++) {
			simMeta.add(simList.get(i)._1);
			simData.add(simList.get(i)._2);
		}
		
		// sort the similarities and the indices
		final Integer[] idx = new Integer[simListSize];
		for (int i = 0; i < simListSize; i++) {
			idx[i] = i;
		}
		Arrays.sort(idx, new Comparator<Integer>() {
		    @Override 
		    public int compare(final Integer o1, final Integer o2) {
		        return Double.compare(simData.get(o1), simData.get(o2));
		    }
		});
		
		// print
		System.out.println("Sorted indices");
		for (int i = 0; i < simListSize; i++) {
			System.out.println(idx[i]);
		}
		
		// sort the List<Vector> of metaData according to the indices
		//TODO
		for (int i = 0; i < simListSize; i++) {
			System.out.println(simMeta.get(idx[i]));
		}
		
	}
	
	
	/**
	 * Main method. Starting place for the execution.
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		
		// check if settings file .prototxt is provided, maybe do this better!
		List<ConfigBaseLayer> globalConfig = null;
		if (args.length == 4) {
		    globalConfig = loadSettings(args[0]);
		} else {
			System.out.print("Usage: spark-submit --class main.java.DeepLearningMain --master local[1] target/DeepManuscriptLearning-0.0.1.jar  <config.prototxt> <test_in1.txt> <test_in2.txt>  <test_out>");
			throw new Exception("Missing command line arguments!");
		}
		
		//TODO add option for train/test/rank in main
		train(globalConfig, args[1], args[2], args[3]);

	}
}
