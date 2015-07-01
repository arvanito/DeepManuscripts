package ch.epfl.ivrl.deepmanuscripts;

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
import ch.epfl.ivrl.deepmanuscripts.DeepModelSettings.*;

import com.google.protobuf.TextFormat;

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
	 * Method that prints a help message of how to run the algorithm.
	 */
	private static void printHelpMessage() {
		System.out.println("Usage: PATH_TO_SPARK_SUBMIT/spark-submit --class " + "ch.epfl.ivrl.deepmanuscripts.DeepLearningMain target/DeepManuscripts-0.1.jar\n");
		System.out.println("Command line args:");
		System.out.println("\t--protobuf=<Input .prototxt file for layer configuration> (file must exist)");
		System.out.println("\t--runningmode=[train,test,rank]");
		System.out.println("\t Train runningmode:");
		System.out.println("\t\t--inputdataset1=<First input file data set> (file must exist)");
		System.out.println("\t\t--inputdataset2=<First input file data set> (file must exist)");
		System.out.println("\t Test runningmode:");
		System.out.println("\t\t--testdataset=<Test dataset> (file must exist)");
		System.out.println("\t Rank runningmode");
		System.out.println("\t\t--test=<Test dataset> (file must exist)");
		System.out.println("\t\t--querydataset=<Query dataset> (file must exist)");
		System.out.println("\t--pathprefix=<Input path prefix for saving the trained model> (can be an empty string)");
		System.out.println("\t--pathprefixtest=<Path prefix for loading the trained model> (files with this prefix must exist)");
		System.out.println("\n");
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
		parser.accepts("help").withOptionalArg();
		parser.accepts("runningmode").withRequiredArg();
		parser.accepts("protobuf").withRequiredArg();
		parser.accepts("inputdataset1").withRequiredArg();
		parser.accepts("inputdataset2").withRequiredArg();
		parser.accepts("querydataset").withRequiredArg();
		parser.accepts("testdataset").withRequiredArg();
		parser.accepts("pathprefix").withOptionalArg();
		parser.accepts("pathprefixtest").withOptionalArg();
		 
		OptionSet options = parser.parse(args);
		
		// first, check if "--help" is given, print out the help and exit
		if (options.hasArgument("help")) {
			printHelpMessage();
			System.exit(0);
		}
		
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
	 * Main method. Three running modes are currently implemented:
	 * 	1. Train
	 * 	2. Test
	 * 	3. Rank
	 * 
	 * @param args Input arguments.
	 * @throws Exception Standard exception. 
	 */
	public static void main(String[] args) throws Exception {
    	
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	sc = new JavaSparkContext(conf);
    	
    	// according to the running mode, run the corresponding method
    	DeepManuscriptConfig config = configure(args);
    	switch (config.getRunningMode()) {
    		case "train":
    			ApplyRunningModes.train(config.getProtoBufConfig(), config.getInputDataset1(), config.getInputDataset2(), config.getPathPrefixTrain(), sc);
    			break;
    		case "test":
    			ApplyRunningModes.test(config.getProtoBufConfig(), config.getTestDataset(), config.getPathPrefixTrain(), config.getPathPrefixTest(), sc);
    			break;
    		case "rank":
    			ApplyRunningModes.rank(config.getProtoBufConfig(), config.getQueryDataset(), config.getTestDataset(), config.getPathPrefixTrain(), config.getPathPrefixTest(), sc);
    			break;
    		default: break;
    	}
	}
	
}
