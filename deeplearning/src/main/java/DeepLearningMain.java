package main.java;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

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
 
    	
		JavaRDD<Vector> input_small_patches = sc.textFile(inputFileSmallPatches).map(new Parse());
		JavaRDD<Vector> input_word_patches = sc.textFile(inputFileLargePatches).map(new Parse());

		// The main loop calls execute() on each of the layers
		JavaRDD<Vector> result = null;
	 	for (int layer_index = 0; layer_index < globalConfig.size(); ++layer_index) {
	 		
	 		// set up the current layer 
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(globalConfig.get(layer_index), layer_index, "x");
			
			// The configLayer has configExtractor only if it convolutional,
			// The multiply Extractor does not need any parameters.
			if (globalConfig.get(layer_index).hasConfigFeatureExtractor()) {
				result = layer.train(input_small_patches, input_word_patches);
			} else {
				result = layer.train(result, result);
			}	
	 	}
		//TODO save also last file
		result.saveAsTextFile(outputFile);
		
		sc.close();
	}
	
	
	public static void test(List<ConfigBaseLayer> globalConfig, String inputFile) {
		
	}
	/*
	 public static void rank() {
	 }
	 */
	
	/**
	 * Main method. Starting place for the execution
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
