package main.java;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import joptsimple.OptionParser;
import joptsimple.OptionSet;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

import com.google.protobuf.TextFormat;

import main.java.DeepModelSettings.*;
import static org.junit.Assert.*;


/**
 * Main class that creates the layer configurations and performs training and testing. 
 * 
 */
public class DeepLearningMain implements Serializable {

	private static final long serialVersionUID = -1047810200264089489L;
	private static JavaSparkContext sc;


	/**
	 * Method that loads the layer configurations from .prototxt file. 
	 * Makes use of the protocol buffers for describing the configuration. 
	 * 
	 * @param prototxt_file Input protocol buffer configuration.
	 * @return List of objects that describe the configuration in each layer.
	 */
	public static List<ConfigBaseLayer> loadSettings(String prototxt_file) {
		
		List<ConfigBaseLayer> globalConfig = null;
		try {
			ConfigManuscripts.Builder builder = ConfigManuscripts.newBuilder();
			
// 			BufferedReader reader = new BufferedReader(new FileReader(prototxt_file));
//			FileInputStream fs = new FileInputStream(prototxt_file);
//			InputStreamReader reader = new InputStreamReader(fs);
//			TextFormat.merge(reader, builder);
			
			// read from HDFS
			FileSystem fs = FileSystem.get(new  Configuration());			
			InputStreamReader reader = new InputStreamReader(fs.open(new Path(prototxt_file)));
			TextFormat.merge(reader, builder);

			// build the settings 
			ConfigManuscripts settings = builder.build();
			
			// get list of configuration layer settings
			globalConfig = settings.getConfigLayerList();			
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
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration.
	 * @param input1 Initial distributed dataset. In the first layer, it will contain small patches. 
	 * For the next layers, it will contain the pooled representations from the previous layer. 
	 * @param input2 Second distributed dataset. In the first layer, it ill contain large patches.
	 * For the next layers, it will be the same as the first input dataset.
	 * @param pathPrefix Path prefix that is used for saving the model.
	 * @throws Exception Standard exception.
	 */
	public static void train(List<ConfigBaseLayer> globalConfig, String input1, String input2, String pathPrefix) throws Exception {
		
		// TODO:: do this more automatic??
		int numPartitions = 400 * 4;
		
		// open the files and convert them to JavaRDD datasets
    	// parse the input datasets to tuples of vectors
		JavaRDD<Tuple2<Vector, Vector>> input1Patches = sc.textFile(input1).map(new ParseTuples()).repartition(numPartitions).cache();
		JavaRDD<Tuple2<Vector, Vector>> input2Patches = sc.textFile(input2).map(new ParseTuples()).repartition(numPartitions).cache();

		// the main loop calls the train method on each of the layers
		JavaRDD<Tuple2<Vector, Vector>> result = null;
	 	for (int layerIndex = 0; layerIndex < globalConfig.size(); ++layerIndex) {
	 		
	 		// set up the current layer
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(globalConfig.get(layerIndex), layerIndex, pathPrefix);
			layer.setSparkContext(sc);
			
			// check is we are in the last layer, if yes, we do not need to perform the last
	 		// feature extraction and pooling
	 		if (layerIndex == globalConfig.size() - 1) {
	 			layer.setNotLast(false);;
	 		} else {
	 			layer.setNotLast(true);
	 		}
	 		
			// the configLayer has configExtractor only if it convolutional,
			// the multiply extractor does not need any parameters.
			if (globalConfig.get(layerIndex).hasConfigFeatureExtractor()) {
				result = layer.train(input1Patches, input2Patches);
			} else {
				result = layer.train(result, result);
			}	
	 	}
		
	 	//result.saveAsTextFile(outputFile);
		sc.close();
	}
	
	
	/**
	 * Main method that tests the trained model. 
	 * 
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration.
	 * @param inputFile Input file that contains the test patches for comparison.
	 * @param pathPrefix Path prefix for loading the trained model.
	 * @param pathPrefixTest Path prefix for saving the test output.
	 * @throws Exception Standard exception.
	 */
	public static void test(List<ConfigBaseLayer> globalConfig, String inputFile, String pathPrefix, String pathPrefixTest) throws Exception {
		
		// TODO:: do this more automatic!!
		int numPartitions = 400*4; 	//Num-workers * cores_per_worker * succesive tasks

		// open the file and convert it to JavaRDD dataset
    	// parse the input dataset to tuples of vectors
		// TODO:: Change this!!
		JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.textFile(inputFile).map(new ParseTuples()).filter(new Function<Tuple2<Vector,Vector>, Boolean>() {
			private static final long serialVersionUID = 2227772572173267004L;

			@Override
			public Boolean call(Tuple2<Vector,Vector> arg0) throws Exception {
				if (arg0._2.size() == 4096){
					return true;
				}
				return false;
			}
		}).repartition(numPartitions);
    	
		// the main loop calls the test method on each of the layers
		JavaRDD<Tuple2<Vector, Vector>> result = null;
	 	for (int layerIndex = 0; layerIndex < globalConfig.size(); layerIndex++) {
	 		
	 		// set up the current layer 
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(globalConfig.get(layerIndex), layerIndex, pathPrefix);
			layer.setSparkContext(sc);
			
			// the configLayer has configExtractor only if it convolutional,
			// the multiply Extractor does not need any parameters.
			if (globalConfig.get(layerIndex).hasConfigFeatureExtractor()) {
				result = layer.test(testPatches);
			} else {
				result = layer.test(result);
			}	
	 	}
	 	
	 	// if specified, save the results
	 	if (pathPrefixTest.length() != 0) {
	 		result.saveAsTextFile(pathPrefixTest + "result");
	 	}
	 	
		sc.close();
	}
	

	/**
	 * Method that performs a word spotting experiments and ranks patches according to their similarity with the input query.
	 * 
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration.
	 * @param queryFile Input file that contains the query's patches.
	 * @param patchesFile Input file that contains the test patches.
	 * @param pathPrefix Path prefix for loading the trained model.
	 * @param pathPrefixTest Path prefix for saving the test output.
	 * @throws Exception Standard exception.
	 */
	public static void rank(List<ConfigBaseLayer> globalConfig, String queryFile, String patchesFile, String pathPrefix, String pathPrefixTest) throws Exception {
	
		//TODO:: do this more automatic
		int numPartitions = 400*4; 	//Num-workers * cores_per_worker * succesive tasks
	
		// load query's patches into a JavaRDD
		// change this!!
		JavaRDD<Tuple2<Vector, Vector>> queryPatches = sc.textFile(queryFile).map(new ParseTuples()).filter(new Function<Tuple2<Vector,Vector>, Boolean>() {
			private static final long serialVersionUID = -5784495184084951735L;

			@Override
			public Boolean call(Tuple2<Vector,Vector> arg0) throws Exception {
				if (arg0._2.size() == 4096){
					return true;
				}
				return false;
			}
		}).repartition(numPartitions);
		
		// load test patches into a JavaRDD
		JavaRDD<Tuple2<Vector, Vector>> imagePatches = sc.textFile(patchesFile).map(new ParseTuples2()).filter(new Function<Tuple2<Vector,Vector>, Boolean>() {
			private static final long serialVersionUID = -6438205114031689814L;

			@Override
			public Boolean call(Tuple2<Vector,Vector> arg0) throws Exception {
				if (arg0._2.size() == 4096){
					return true;
				}
				return false;
			}
		}).repartition(numPartitions);
	
		// the main loop calls test() on each of the layers
		JavaRDD<Tuple2<Vector, Vector>> resultQuery = null;
		JavaRDD<Tuple2<Vector, Vector>> resultImage = null;
	 	for (int layerIndex = 0; layerIndex < globalConfig.size(); layerIndex++) {
	 		
	 		// set up the current layer 
			DeepLearningLayer layer = BaseLayerFactory.createBaseLayer(globalConfig.get(layerIndex), layerIndex, pathPrefix);
			layer.setSparkContext(sc);
			
			// the configLayer has configExtractor only if it convolutional,
			// the multiply Extractor does not need any parameters.
			if (globalConfig.get(layerIndex).hasConfigFeatureExtractor()) {
				
				// query and test patches
				resultQuery = layer.test(queryPatches);
				resultImage = layer.test(imagePatches);
			} else {
				
				// query and test patches
				resultQuery = layer.test(resultQuery);
				resultImage = layer.test(resultImage);
			}	
	 	}

	 	// if specified, save the two test outputs
	 	if (pathPrefixTest.length() != 0) {
	 		resultQuery.saveAsTextFile(pathPrefixTest + "resultTest");
	 		resultImage.saveAsTextFile(pathPrefixTest + "resultImage");
	 	}
	 	
	 	Iterator<Tuple2<Vector, Vector>> queryPatchesList = resultQuery.collect().iterator();
	 	
	 	// for each query patch, compute its most similar one from the test image patches
	 	int testPatch = 0;
	 	while(queryPatchesList.hasNext()) {
	 		testPatch++;
	 		
	 		// next query patch
	 		Tuple2<Vector,Vector> querryPair = queryPatchesList.next();
	 		
	 		// compute similarity between the current query patch and all test patches
	 		ComputeSimilarityPair compSim = new ComputeSimilarityPair(querryPair);
	 		List<Tuple2<Vector, Double>> sim = resultImage.map(compSim).map(new Function<Tuple2<Vector,Double>, Tuple2<Vector,Double>>() {
				private static final long serialVersionUID = -566603841854522173L;
	
				@Override
				public Tuple2<Vector, Double> call(Tuple2<Vector, Double> pair) throws Exception {
					return new Tuple2<Vector,Double> (pair._1,-pair._2);
				}
			}).takeOrdered(300, new VectorComparator());	//TODO:: change this!!!
 		
	 		// if specified, save the similarities
	 		if (pathPrefixTest.length() != 0) {
	 			sc.parallelize(sim).saveAsTextFile(pathPrefixTest + "_similarities_" + testPatch);
	 		}
	 	}
 	
	 	sc.close();
	}
	
	
	/**
	 * Main method for ranking the candidate patches with the query. 
	 * 
	 * @param globalConfig List of ConfigBaseLayer objects that represent the current configuration
	 * @param inputFileQuery Input file that contains the query
	 * @param inputFilePatches Input file that contains the candidate patches
	 */
//	public static void rank(List<ConfigBaseLayer> globalConfig, String inputFileQuery, String inputFilePatches) throws Exception {
//		
//		// open the test file and convert it to a JavaRDD<Vector> dataset
//		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
//    	JavaSparkContext sc = new JavaSparkContext(conf);
//    	
//		// load query 
//    	// TODO make it work for multiple queries at the same time, do a loop over the queries
//    	// The same procedure of the test patches should apply to the queries
//		JavaRDD<Tuple2<Vector, Vector>> query = sc.textFile(inputFileQuery).map(new ParseTuples());
//		Tuple2<Vector, Vector> queryTuple = query.collect().get(0);
//		//JavaRDD<Tuple2<Vector, Vector>> query = sc.objectFile(inputFileQuery);
//    	
//		// load candidate patches
//		JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.textFile(inputFilePatches).map(new ParseTuples());
//		//JavaRDD<Tuple2<Vector, Vector>> testPatches = sc.objectFile(inputFilePatches);
//		
//		// get the dimensions of the large learned patches
//		int[] inputDims = {globalConfig.get(0).getConfigFeatureExtractor().getInputDim1(), globalConfig.get(0).getConfigFeatureExtractor().getInputDim2()};
//		
//		// extract (overlapping?) patches from the input query and candidate patches
//		// TODO Check this!
//		Vector vecSize = queryTuple._1;
//		int[] vecSizeInt = new int[vecSize.size()];
//		for (int i = 0; i < vecSize.size(); i++) {
//			vecSizeInt[i] = (int) vecSize.apply(i);
//		}
//		JavaRDD<Tuple2<Vector, Vector>> queryPatches = query.flatMap(new ExtractPatches(vecSizeInt, inputDims));
//		testPatches = testPatches.flatMap(new ExtractPatches(vecSizeInt, inputDims));
//		
//		// compute representations for the query and candidate patches
//		JavaRDD<Tuple2<Vector, Vector>> queryReps = testRDD(globalConfig, queryPatches);
//		JavaRDD<Tuple2<Vector, Vector>> testReps = testRDD(globalConfig, testPatches);
//		
//		// sort by key the query and test patches representations
//		Comparator<Vector> compareVectors = new CompareVectors();
//		JavaPairRDD<Vector, Vector> queryPairs = JavaPairRDD.fromJavaRDD(queryReps).sortByKey(compareVectors);
//		JavaPairRDD<Vector, Vector> testPairs = JavaPairRDD.fromJavaRDD(testReps).sortByKey(compareVectors);
//		
//		// concatenate patch representations to represent the original query
//		List<Tuple2<Vector, Vector>> queryRepList = queryPairs.collect();
//		int queryRepLength = queryRepList.size();			// size of the list
//		int vecLength = queryRepList.get(0)._2.size();		// size of the patch representations in the list
//		double[] queryData = new double[queryRepLength*vecLength];
//
//		int k = 0;	// offset for each Vector in the List
//		for (int i = 0; i < queryRepLength; i++) {
//			System.arraycopy(queryRepList.get(i)._2, 0, queryData, k, vecLength);
//			k = k + vecLength;
//		}
//		Tuple2<Vector, Vector> queryRep = new Tuple2<Vector, Vector>(queryTuple._1, Vectors.dense(queryData));
//		
//		// concatenate patch representations from the RDD
//		// TODO Correct this !!!!!!!!
//		testPairs = testPairs.reduceByKey(
//				new Function2<Vector, Vector, Vector>() {
//					private static final long serialVersionUID = 5851620097719920872L;
//
//					// concatenate two Vectors with the same key
//					public Vector call(Vector v1, Vector v2) {
//						double[] out = new double[v1.size() + v2.size()];
//
//						System.arraycopy(v1.toArray(), 0, out, 0, v1.size());
//						System.arraycopy(v2.toArray(), 0, out, v1.size(), v2.size());
//
//						return Vectors.dense(out);
//					}
//				}
//			);
//		testReps = testPairs.rdd().toJavaRDD();	// convert back to JavaRDD<Tuple2<Vector, Vector>>
//		
//		// compute cosine similarities between the query representation and the candidate patches' representations
//		JavaRDD<Tuple2<Vector, Double>> cosineSim = testReps.map(new ComputeSimilarityPair(queryRep));
//		
//		/****************** rank the results ******************/
//		List<Tuple2<Vector, Double>> simList = cosineSim.collect();
//		int simListSize = simList.size();
//		
//		// extract in separate Lists the Vector metadata and Double similarities
//		final List<Vector> simMeta = new ArrayList<Vector>(simListSize);
//		final List<Double> simData = new ArrayList<Double>(simListSize);
//		for (int i = 0; i < simListSize; i++) {
//			simMeta.add(simList.get(i)._1);
//			simData.add(simList.get(i)._2);
//		}
//		
//		// sort the similarities and the indices
//		final Integer[] idx = new Integer[simListSize];
//		for (int i = 0; i < simListSize; i++) {
//			idx[i] = i;
//		}
//		Arrays.sort(idx, new Comparator<Integer>() {
//		    @Override 
//		    public int compare(final Integer o1, final Integer o2) {
//		        return Double.compare(simData.get(o1), simData.get(o2));
//		    }
//		});
//		
//		// print
//		System.out.println("Sorted indices");
//		for (int i = 0; i < simListSize; i++) {
//			System.out.println(idx[i]);
//		}
//		
//		// sort the List<Vector> of metaData according to the indices
//		for (int i = 0; i < simListSize; i++) {
//			System.out.println(simMeta.get(idx[i]));
//		}
//		/**************** finished ranking **********************/
//		
//		sc.close();
//	}
	
	

	/**
	 * Method that checks if a required argument is provided in the command line.
	 * 
	 * @param options Options parsed.
	 * @param necessaryFlag The required option.
	 * @throws IllegalArgumentException
	 */
	private static void checkArgIsProvided(OptionSet options, String necessaryFlag) throws IllegalArgumentException {
		 if (!options.hasArgument(necessaryFlag)) 
			 throw new IllegalArgumentException("Missing necessary flag " + necessaryFlag);
	}

	
	/**
	 * Method that checks if the required option has a valid value.
	 * 
	 * @param value The input value from the command line.
	 * @param l The list of valid values for the option.
	 * @throws IllegalArgumentException
	 */
	private static void checkArgValueIsInList(String value, List<String> l) throws IllegalArgumentException {
		if (l.contains(value) == false) 
			throw new IllegalArgumentException("Invalid value " + value); // TODO Print possible values
	}
	
	
	/**
	 * Method that parses the input arguments from the command line and creates the necessary configuration objects.
	 * 
	 * @param args Input array of parameters from the command line.
	 * @return A configuration object.
	 */
	private static DeepManuscriptConfig configure(String[] args) {
		
		// create the option parser and make it accept the input options
		OptionParser parser = new OptionParser();
		parser.accepts("help");
		parser.accepts("runningmode").withRequiredArg();
		parser.accepts("protobuf").withRequiredArg();
		parser.accepts("inputdataset1").withRequiredArg();
		parser.accepts("inputdataset2").withRequiredArg();
		parser.accepts("querydataset").withRequiredArg();
		parser.accepts("testdataset").withRequiredArg();
		parser.accepts("pathprefix").withOptionalArg();
		parser.accepts("pathprefixtest").withOptionalArg();
		 
		OptionSet options = parser.parse(args);
		
		// the options "runningmode" and "protobuf" are required
		checkArgIsProvided(options, "runningmode");
		checkArgIsProvided(options, "protobuf");
		
		// check valid string values for the "runningmode" option
		String runningMode = (String) options.valueOf("runningmode"); 
		checkArgValueIsInList(runningMode,
	    	new ArrayList<String>(Arrays.asList("train","test","rank"))
	    );
		
		// check if the protobuf file exists
	    String protoBufFile = (String) options.valueOf("protobuf");
	    File f = new File(protoBufFile);
	    if(!f.exists() || f.isDirectory()) {
	    	throw new IllegalArgumentException("Protobuf file does not exist or is a directory."); 
	    }
	    
	    // load protobuf configuration
	    List<ConfigBaseLayer> protoBufConfig = loadSettings(protoBufFile);
	    
		// check if we save the trained model
		//TODO:: check if pathprefix is part of some directories
		String pathPrefix = (String) options.valueOf("pathprefix");
		if (pathPrefix == null) {
			pathPrefix = "";
		}
        
		// check if we save the test output
		String pathPrefixTest = (String) options.valueOf("pathprefixtest");
		if (pathPrefixTest == null) {
			pathPrefixTest = "";
		}
		
	    // depending on the running mode, we either train, test of rank
		DeepManuscriptConfig config;
	    switch (runningMode) {
        	case "train":
        		// in the "train" running mode, we require two datasets
        		checkArgIsProvided(options, "inputdataset1");
        		checkArgIsProvided(options, "inputdataset2");
        		String inputDataset1 = (String) options.valueOf("inputdataset1");
        		String inputDataset2 = (String) options.valueOf("inputdataset2");
        		config = new DeepManuscriptConfig(runningMode, protoBufConfig, inputDataset1, inputDataset2, null, null, pathPrefix, pathPrefixTest);
        		break;
        	case "test":
        		// in the "test" mode, we require an input pathprefix and an input dataset
        		checkArgIsProvided(options, "testdataset");
        		String testFileTest = (String) options.valueOf("testdataset");
        		config = new DeepManuscriptConfig(runningMode, protoBufConfig, null, null, null, testFileTest, pathPrefix, pathPrefixTest);
        		break;
        	case "rank":
        		// in the "rank" running mode, we have the same arguments as in the test, but 
        		// instead of one test file, we have two: one for the query(s) and another for the test patches
        		checkArgIsProvided(options, "querydataset");
        		checkArgIsProvided(options, "testdataset");
        		String queryFile = (String) options.valueOf("querydataset");
        		String testFileRank = (String) options.valueOf("testdataset");
        		config = new DeepManuscriptConfig(runningMode, protoBufConfig, null, null, queryFile, testFileRank, pathPrefix, pathPrefixTest);
        		break;
         
        	default: 
        		config = null;
        		break;
	    }
	    
	    return config;
	}
	
	
	/**
	 * Main method.
	 * 
	 * @param args Input arguments.
	 * @throws Exception Standard exception. 
	 */
	public static void main(String[] args) throws Exception {
		 
		SparkConf conf = new SparkConf().setAppName("DeepManuscript testing");
    	sc = new JavaSparkContext(conf);
    	
    	// according to the running mode, run the corresponding method
    	DeepManuscriptConfig config = configure(args);
    	switch (config.getRunningMode()) {
    		case "train":
    			train(config.getProtoBufConfig(), config.getInputDataset1(), config.getInputDataset2(), config.getPathPrefixTrain());
    			break;
    		case "test":
    			test(config.getProtoBufConfig(), config.getTestDataset(), config.getPathPrefixTrain(), config.getPathPrefixTest());
    			break;
    		case "rank":
    			rank(config.getProtoBufConfig(), config.getQueryDataset(), config.getTestDataset(), config.getPathPrefixTrain(), config.getPathPrefixTest());
    			break;
    		default: break;
    	}
	}
	
}
