package main.java;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

import com.google.protobuf.TextFormat;

import main.java.ConvolutionalNeuralNetworkSettings.CNNSettings;
import main.java.ConvolutionalNeuralNetworkSettings.ConvolutionalLayer;

public class DeepLearningMain {
	public static void loadSettings(String prototxt_file) {
		// TODO return the settings structure
		try {
			CNNSettings.Builder builder = CNNSettings.newBuilder();
			//BufferedReader reader = new BufferedReader(new FileReader(prototxt_file));
			FileInputStream fs = new FileInputStream(prototxt_file);
			InputStreamReader reader = new InputStreamReader(fs);
			TextFormat.merge(reader, builder);
			
			// Settings file created
			CNNSettings settings = builder.build();
			if (settings.hasLearningRate()) {
				System.out.printf("learning rate is %f\n", settings.getLearningRate());
			}
			System.out.printf("nbr conv layers %d\n", settings.getConvLayerCount());
			for (ConvolutionalLayer convlayer: settings.getConvLayerList()) {
				System.out.printf("Num filters %d, Filter size %d\n", 
						convlayer.getNumFilters(), convlayer.getFilterW());
			}
			
			reader.close();
		} catch (IOException e) {
			System.err.println("Input .prototxt file not found");
			System.err.println(prototxt_file);
			e.printStackTrace();
			System.exit(1);
		}
	}
	public static void main(String[] args) {
		if (args.length > 2) {
			// settings file .prototxt provided
		    loadSettings(args[2]);
		}
		
		SparkConf conf = new SparkConf().setAppName("DeepManuscript learning");
    	JavaSparkContext sc = new JavaSparkContext(conf);
    	String inputFile = args[0];
    	String outputFile = args[1];
		JavaRDD<Vector> data = sc.textFile(inputFile).map(new Parse());
		
		// The main loop could loop over this kind of stuff (and handle the saving of features and so on)
		DeepLearningLayer layer1 = new DummyLayer();
		JavaRDD<Vector> features = layer1.learnFeatures(data);
		JavaRDD<Vector> represent = layer1.extractFeatures(data, features);
		JavaRDD<Vector> pooled = layer1.pool(represent);
		
		pooled.saveAsTextFile(outputFile);
		sc.close();
	}
}
