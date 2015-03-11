package main.java;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

public class DeepLearningMain {
	public static void main(String[] args) {
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
