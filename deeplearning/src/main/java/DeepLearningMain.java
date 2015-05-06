package main.java;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;
import scala.reflect.ClassTag;

import com.google.protobuf.TextFormat;

import main.java.DeepModelSettings.*;

/**
 * Main class.
 * 
 */
public class DeepLearningMain {
	
	private static JavaSparkContext sc;


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
//		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
//    	JavaSparkContext sc = new JavaSparkContext(conf);
 
		JavaRDD<Vector> inputSmallPatches = sc.textFile(inputFileSmallPatches).map(new Parse());
		JavaRDD<Vector> inputWordPatches = sc.textFile(inputFileLargePatches).map(new Parse());

		//testMe(inputSmallPatches);
		
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
	
	
	private static class shareMe implements Serializable{


		/**
		 * 
		 */
		private static final long serialVersionUID = -4709600624146273244L;
		public int[] val;
		public shareMe(int[] i){
			this.val = i;
		}
	}
	
	private static class runMeBr implements Function<Vector, Integer>{
		

		/**
		 * 
		 */
		private static final long serialVersionUID = -8025122885549295898L;
		private Broadcast<shareMe> bObj;
		
		public runMeBr(Broadcast<shareMe> bObj) {
			this.bObj = bObj;
		}

		@Override
		public Integer call(Vector arg0) throws Exception {
			return  bObj.value().val[0];
		}
		
	}
	
	private static class runMe implements Function<Vector, Integer>{
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -354064928874130047L;
		private shareMe s;
		
		public runMe(shareMe s) {
			this.s = s;
		}

		@Override
		public Integer call(Vector arg0) throws Exception {
			return s.val[0];
		}
		
	}
	
	private static void testMe(JavaRDD<Vector> data) throws InterruptedException {
		int[] longArray = new int[300000];
		longArray[0] = 3;
		
		long i = data.count();
		
//		long t1 = System.currentTimeMillis();
		shareMe shareMe = new shareMe(longArray);
		runMe runMeFct = new runMe(shareMe);
		JavaRDD<Integer> data1 = data.map(runMeFct);
		System.out.println("\n-------------------\n"+data1.collect().iterator().next()+"\n-------------------\n");
//		long t2 = System.currentTimeMillis();
		
//		for (int i=0;i<10000;i++){
//			System.out.print(""+longArray[0]);
//		}
		
//		long t3 = System.currentTimeMillis();
		shareMe shareMe2 = new shareMe(longArray);
		Broadcast<shareMe> shareMeBr =  sc.broadcast(shareMe2);//, scala.reflect.ClassTag$.MODULE$.apply(shareMe.class));
		runMeBr runMeFctBr = new runMeBr(shareMeBr);
		JavaRDD<Integer> data2 = data.map(runMeFctBr);
		System.out.println("\n-------------------\n"+data2.collect().iterator().next()+"\n-------------------\n");
//		long t4 = System.currentTimeMillis();
//		
//		System.out.println("\n-------------------\n"+(t2-t1)+" "+(t4-t3)+"\n-------------------\n");
		
		Thread.sleep(180000);
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
//		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
//    	JavaSparkContext sc = new JavaSparkContext(conf);
    	
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
	public static JavaRDD<Vector> testRDD(List<ConfigBaseLayer> globalConfig, JavaRDD<Vector> testPatches) throws Exception {
		
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
	public static JavaRDD<Vector> testRDDTuple(List<ConfigBaseLayer> globalConfig, JavaRDD<Tuple2<Vector, Vector>> testPatches) throws Exception {
		
		// The main loop calls test() on each of the layers
		JavaRDD<Vector> result = null;
	 	for (int layerIndex = 0; layerIndex < globalConfig.size(); ++layerIndex) {
	 		
	 		// set up the current layer 
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(globalConfig.get(layerIndex), layerIndex, "x");
			
			// The configLayer has configExtractor only if it convolutional,
			// The multiply Extractor does not need any parameters.
			if (globalConfig.get(layerIndex).hasConfigFeatureExtractor()) {
			//	result = layer.test(testPatches);
			} else {
			//	result = layer.test(result);
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
//		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
//    	sc = new JavaSparkContext(conf);
    	
		// load query 
		JavaRDD<Tuple2<Vector, Vector>> query = sc.textFile(inputFileQuery).map(new ParseTuples());
		Tuple2<Vector, Vector> queryTuple = query.collect().get(0);
		//JavaRDD<Tuple2<Vector, Vector>> query = sc.objectFile(inputFileQuery);
    	
		// load candidate patches
		JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.textFile(inputFilePatches).map(new ParseTuples());
		//JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.objectFile(inputFilePatches);
		
		// union of the two JavaRDD<Vector> datasets
		//JavaRDD<Vector> completePatches = queryRDD.union(testPatches);
		
		// get the dimensions of the large learned patches
		int[] inputDims = {globalConfig.get(0).getConfigFeatureExtractor().getInputDim1(), globalConfig.get(0).getConfigFeatureExtractor().getInputDim2()};
		
		// extract (overlapping?) patches from the input query and candidate patches
		Vector vecSize = queryTuple._1;
		int[] vecSizeInt = new int[vecSize.size()];
		for (int i = 0; i < vecSize.size(); i++) {
			vecSizeInt[i] = (int) vecSize.apply(i);
		}
		//JavaRDD<Tuple2<Vector, Vector>> queryPatches = query.flatMap(new ExtractPatchesTuples(vecSizeInt, inputDims));
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
		
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	sc = new JavaSparkContext(conf);
		
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
