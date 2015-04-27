package main.java;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 *  The class offers different ways of storing DenseMatrix and Vector from linalg package to hdfs.
 *  The objects can be saved either as text files or as objects.
 * 
 * @author Viviana Petrescu
 *
 */

public class LinAlgebraIOUtils {
	
	/**
	 * Saves a Vector as a text file.
	 * 
	 * @param input
	 * @param outFile
	 * @param sc
	 */
	public static void saveVectorToText(Vector input, String outFile, JavaSparkContext sc) {
		List<Vector> temp_input = new ArrayList<Vector>();
		temp_input.add(input);
		
		// Transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsTextFile(outFile);
	}
	
	
	/**
	 * Saves a Vector as an object.
	 * 
	 * @param input
	 * @param outFile
	 * @param sc
	 */
	public static void saveVectorToObject(Vector input, String outFile, JavaSparkContext sc) {
		List<Vector> temp_input = new ArrayList<Vector>();
		temp_input.add(input);
		
		// Transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsObjectFile(outFile);
	}
	
	
	/**
	 * Loads a Vector from a text file.
	 * 
	 * @param inFile
	 * @param sc
	 * @return
	 */
	public static Vector loadVectorFromText(String inFile, JavaSparkContext sc) {
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
	
	
	/**
	 * Loads a Vector from an object file.
	 * 
	 * @param inFile
	 * @param sc
	 * @return
	 */
	public static Vector loadVectorFromObject(String inFile, JavaSparkContext sc) {
		JavaRDD<Vector> out = sc.objectFile(inFile);
		List<Vector> a = out.collect();
		return Vectors.dense(a.get(0).toArray());
	}
	
	
	/**
	 * !!!
	 * @param input
	 * @param outFile
	 * @param sc
	 */
	public static void saveVectorToText(Matrix input, String outFile, JavaSparkContext sc) {
		List<Vector> temp_zca = new ArrayList<Vector>();
		for (int i = 0; i < input.numRows();++i) {
			double[] darray = new double[input.numCols()];
			for (int j = 0; j < input.numCols(); ++j) {
				darray[j] = input.apply(i, j);
			}
			Vector row = Vectors.dense(darray);
			temp_zca.add(row);
		}
		sc.parallelize(temp_zca).saveAsTextFile(outFile);
	}
	
	
	/**
	 * Loads a Matrix from a text file.
	 * 
	 * @param inFile
	 * @param sc
	 * @return
	 */
	//TODO check row or column major
	public static Matrix loadMatrixFromText(String inFile, JavaSparkContext sc) {
		// Read back the file as an array of strings
		JavaRDD<String> in_read = sc.textFile(inFile);

		List<String> in_string = in_read.collect();
		// Since it is a vector, it has 1 dimension equal to 1
		int nCols = in_string.size();
		String m = in_string.get(0);
		m = m.substring(1, m.length()-2);
		int nRows = m.split(",").length;
		double[] out_vector = new double[nRows * nCols];
		int idx = 0;
		for (int i = 0; i < in_string.size(); ++i) {
			String m2 = in_string.get(i);
			m2 = m2.substring(1, m.length()-2);
			String[] parts = m.split(",");
			int vector_size = parts.length;
			for (int j = 0; j < vector_size; ++j) {
				out_vector[idx++] = Double.parseDouble(parts[j]);
			}
		}
		return Matrices.dense(nRows, nCols, out_vector);
	}
	
	
	/**
	 * Saves a Matrix to a text file.
	 * 
	 * @param input
	 * @param outFile
	 * @param sc
	 */
	//TODO check row or column major
	public static void saveMatrixToText(Matrix input, String outFile, JavaSparkContext sc) {
		List<Vector> temp_input = new ArrayList<Vector>();
		for (int i = 0; i < input.numCols(); ++i) {
			double[] temp = new double[input.numRows()];
			for (int j = 0; j < input.numRows(); ++j) {
				temp[j] = input.apply(j, i);
			} 
			Vector col_vector = Vectors.dense(temp);
			temp_input.add(col_vector);
		}
		// Transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsTextFile(outFile);
	}
	
	
	/**
	 * Loads a Matrix from an object file.
	 * 
	 * @param inFile
	 * @param sc
	 * @return
	 */
	// Load Column major matrix
	public static Matrix loadMatrixFromObject(String inFile, JavaSparkContext sc) {
		JavaRDD<Vector> input = sc.objectFile(inFile);
		List<Vector> col_matrix = input.collect();
		int nCols = col_matrix.size();
		int nRows = col_matrix.get(0).size();
		double[] temp = new double[nRows*nCols];
		int idx = 0;
		for (int i = 0; i < nCols; ++i) {
			for (int j = 0; j < nRows; ++j) {
				temp[idx++] = col_matrix.get(i).apply(j);
			}
		}
		return Matrices.dense(nRows, nCols, temp);
	}
	
	
	/**
	 * Saves a Matrix from an object file.
	 * 
	 * @param input
	 * @param outFile
	 * @param sc
	 */
	// Save the matrix column major
	public static void saveMatrixToObject(Matrix input, String outFile, JavaSparkContext sc) {
		List<Vector> temp_input = new ArrayList<Vector>();
		for (int i = 0; i < input.numCols(); ++i) {
			double[] temp = new double[input.numRows()];
			for (int j = 0; j < input.numRows(); ++j) {
				temp[j] = input.apply(j, i);
			} 
			Vector col_vector = Vectors.dense(temp);
			temp_input.add(col_vector);
		}
		// Transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsObjectFile(outFile);
	}
	
	
	/**
	 * Saves an array of Vectors to an object file.
	 * 
	 * @param input
	 * @param outFile
	 * @param sc
	 */
	public static void saveVectorArrayToObject(Vector[] input, String outFile, JavaSparkContext sc) {
		List<Vector> temp_input = new ArrayList<Vector>();
		for (int i = 0; i < input.length; ++i)
			temp_input.add(input[i]);
		// Transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsObjectFile(outFile);
	}
	
	
	/**
	 * Loads an array of Vectors from an object file.
	 * 
	 * @param inFile
	 * @param sc
	 * @return
	 */
	public static Vector[] loadVectorArrayFromObject(String inFile, JavaSparkContext sc) {
		JavaRDD<Vector> out = sc.objectFile(inFile);
		List<Vector> a = out.collect();
		Vector[] output = new Vector[a.size()];
		for (int i = 0; i < a.size(); ++i)
			output[i] = Vectors.dense(a.get(i).toArray());
		return output;
	}
}
