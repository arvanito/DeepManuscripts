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

import main.java.DeepModelSettings.ConfigBaseLayer;
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
	 * @param prototxt_file
	 * @return 
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
			System.err.println("Input .prototxt file not found");
			System.err.println(prototxt_file);
			e.printStackTrace();
			System.exit(1);
		}

		return globalConfig;
	}
	public static void main(String[] args) throws Exception {
		
		// check if settings file .prototxt is provided, maybe do this better!
		List<ConfigBaseLayer> globalConfig = null;
		if (args.length > 2) {
		    globalConfig = loadSettings(args[2]);
		} else {
			throw new Exception("No .prototxt file found!");
		}
		
		// open files and convert them to JavaRDD<Vector> datasets
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
    	String inputFile = args[0];
    	String outputFile = args[1];
		JavaRDD<Vector> data = sc.textFile(inputFile).map(new Parse());
		
		// The main loop calls execute() on each of the layers
		DeepLearningLayer layer1 = new DummyLayer(globalConfig.get(0));
		JavaRDD<Vector> result = layer1.execute(data);
		result.saveAsTextFile(outputFile);
		
		sc.close();
	}
}
