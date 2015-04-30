package main.java;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
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
 
		JavaRDD<Vector> inputSmallPatches = sc.textFile(inputFileSmallPatches).map(new Parse());
		JavaRDD<Vector> inputWordPatches = sc.textFile(inputFileLargePatches).map(new Parse());

		// The main loop calls train() on each of the layers
		JavaRDD<Vector> result = null;
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
    	
		JavaRDD<Vector> testPatches = sc.textFile(inputFile).map(new Parse());
    	
		// The main loop calls test() on each of the layers
		JavaRDD<Vector> result = null;
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
	 * @param testPatches JavaRDD<Vector> which contains all the candidate patches
	 * @return The resulting representations of the test patches
	 */
	public static JavaRDD<Vector> test(List<ConfigBaseLayer> globalConfig, JavaRDD<Vector> testPatches) throws Exception {
		
		// The main loop calls test() on each of the layers
		JavaRDD<Vector> result = null;
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
	 * Main method for testing the trained model. 
	 * 
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration
	 * @param testPatches JavaRDD<Tuple2<Vector, Vector>> which contains all the candidate patches
	 * @return The resulting representations of the test patches
	 */
	public static JavaRDD<Vector> test(List<ConfigBaseLayer> globalConfig, JavaRDD<Tuple2<Vector, Vector>> testPatches) throws Exception {
		
		// The main loop calls test() on each of the layers
		JavaRDD<Vector> result = null;
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
	 * Main method for ranking the candidate patches with the query 
	 * 
	 * @param globalConfig
	 * @param inputFileQuery
	 * @param inputFilePatches
	 */
	public static void rank(List<ConfigBaseLayer> globalConfig, String inputFileQuery, String inputFilePatches) throws Exception {
		
		// open the test file and convert it to a JavaRDD<Vector> dataset
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
    	
		// load query 
		//JavaRDD<Tuple2<Vector, Vector>> query = sc.textFile(inputFileQuery).map(new Parse());
    	//JavaRDD<Tuple2<Vector, Vector>> query = sc.objectFile(inputFileQuery);
    	
		// load candidate patches
		//JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.textFile(inputFilePatches).map(new Parse());
		//JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.objectFile(inputFilePatches);
		
		// union of the two JavaRDD<Vector> datasets
		//JavaRDD<Vector> completePatches = queryRDD.union(testPatches);
		
		// get the dimensions of the large learned patches
		int[] inputDims = {globalConfig.get(0).getConfigFeatureExtractor().getInputDim1(), globalConfig.get(0).getConfigFeatureExtractor().getInputDim2()};
		
		// extract (overlapping?) patches from the input query and candidate patches
		int[] vecSize = {2,2};
		int[] patchSize = {2,2};
		//JavaRDD<Tuple2<Vector, Vector>> queryPatches = query.flatMap(new ExtractPatchesTuples(vecSize, patchSize));
		//testPatches = testPatches.flatMap(new ExtractPatchesTuples(vecSize, patchSize));
		
		//TODO:: Change the test method to take arguments of JavaRDD<Tuple2<Vector,Vector>>
		
		// compute representations for the query and candidate patches
		//JavaRDD<Vector> queryRepresentation = test(globalConfig, queryPatches);
		//JavaRDD<Vector> testRepresentation = test(globalConfig, testPatches);
		
		// concatenate patch representations to represent the original big candidate patch
		//List<Vector> queryRepList = queryRepresentation.collect();
		//String queryString = queryRepList.toString();

		// compute cosine similarities between the query representation and the condidate patches' representations
		//JavaRDD<Double> cosineSim = testRepresentation.map(new ComputeSimilarity(queryV));
		
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
