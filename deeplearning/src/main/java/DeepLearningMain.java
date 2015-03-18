package main.java;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

import com.google.protobuf.TextFormat;

import main.java.DeepModelSettings.ManuscriptsConfig;
import main.java.DeepModelSettings.BaseLayer;

/**
 * Main class.
 * 
 */
public class DeepLearningMain {
	public static void loadSettings(String prototxt_file) {
		// TODO return the settings structure
		try {
			ManuscriptsConfig.Builder builder = ManuscriptsConfig.newBuilder();
			//BufferedReader reader = new BufferedReader(new FileReader(prototxt_file));
			FileInputStream fs = new FileInputStream(prototxt_file);
			InputStreamReader reader = new InputStreamReader(fs);
			TextFormat.merge(reader, builder);
			
			// Settings file created
			ManuscriptsConfig settings = builder.build();
			System.out.printf("# base layers is %d\n", settings.getBaseLayerCount());
			
			for (BaseLayer blayer: settings.getBaseLayerList()) {
			}
			
			reader.close();
		} catch (IOException e) {
			System.err.println("Input .prototxt file not found");
			System.err.println(prototxt_file);
			e.printStackTrace();
			System.exit(1);
		}
	}
	public static void main(String[] args) throws Exception {
		if (args.length > 2) {
			// settings file .prototxt provided
		    loadSettings(args[2]);
		}
		
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
    	String inputFile = args[0];
    	String outputFile = args[1];
		JavaRDD<Vector> data = sc.textFile(inputFile).map(new Parse());
		
		// The main loop calls execute() on each of the layers
		DeepLearningLayer layer1 = new DummyLayer();
		JavaRDD<Vector> result = layer1.execute(data);
		result.saveAsTextFile(outputFile);
		
		sc.close();
	}
}
