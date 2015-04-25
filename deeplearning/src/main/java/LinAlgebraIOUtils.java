package main.java;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 *  The class offers different ways of storing DenseMatrix and Vector from linalg package to hdfs.
 *  The objects can be saved either as text file or as objects.
 * 
 * @author Viviana Petrescu
 *
 */

public class LinAlgebraIOUtils {
	public static void saveToText(Vector input, String outFile, JavaSparkContext sc) {
		List<Vector> temp_input = new ArrayList<Vector>();
		temp_input.add(input);
		// Transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsTextFile(outFile);
	}
	public static void saveToObject(Vector input, String outFile, JavaSparkContext sc) {
		List<Vector> temp_input = new ArrayList<Vector>();
		temp_input.add(input);
		// Transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsObjectFile(outFile);
	}
	public static Vector loadFromText(String inFile, JavaSparkContext sc) {
		// Read back the file as an array of strings
		JavaRDD<String> in_read = sc.textFile(inFile);

		List<String> in_string = in_read.collect();
		// Since it is a vector, it has 1 dimension equal to 1
		assert(in_string.size() == 1);
		System.out.println(in_string);
		String m = in_string.get(0);
		m = m.substring(1, m.length()-2);
		String[] parts = m.split(",");
		int vector_size = parts.length;
		double out_vector[] = new double[vector_size];
		for (int i = 0; i < vector_size; ++i) {
			out_vector[i] = Double.parseDouble(parts[i]);
		}
		return Vectors.dense(out_vector);
	}
	public static Vector loadFromObject(String inFile, JavaSparkContext sc) {
		JavaRDD<Vector> out = sc.objectFile(inFile);
		List<Vector> a = out.collect();
		return Vectors.dense(a.get(0).toArray());
	}
	
}
