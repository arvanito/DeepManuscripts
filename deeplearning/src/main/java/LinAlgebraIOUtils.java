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
	 * Loads a Vector from an object file.
	 * 
	 * @param inFile Input file to load.
	 * @param sc Spark context.
	 * @return Loaded vector.
	 */
	public static Vector loadVectorFromObject(String inFile, JavaSparkContext sc) {
		
		// read the object and collect
		JavaRDD<Vector> out = sc.objectFile(inFile);
		List<Vector> a = out.collect();
		
		return Vectors.dense(a.get(0).toArray());
	}
	
	
	/**
	 * Loads a Matrix from an object file column by column.
	 * 
	 * @param inFile Input file to load.
	 * @param sc Spark context.
	 * @return Column-major matrix.
	 */
	//TODO:: do this row-wise.
	public static Matrix loadMatrixFromObject(String inFile, JavaSparkContext sc) {
		
		// load the file as a JavaRDD and collect
		JavaRDD<Vector> input = sc.objectFile(inFile);
		List<Vector> col_matrix = input.collect();
		
		// create the column-major matrix
		int nCols = col_matrix.size();
		int nRows = col_matrix.get(0).size();
		double[] temp = new double[nRows*nCols];
		int idx = 0;
		for (int i = 0; i < nCols; i++) {
			for (int j = 0; j < nRows; j++) {
				temp[idx++] = col_matrix.get(i).apply(j);
			}
		}
		
		return Matrices.dense(nRows, nCols, temp);
	}
	
	
	
	/**
	 * Loads a Vector from a text file.
	 * 
	 * @param inFile Input file to load.
	 * @param sc Spark context.
	 * @return Loaded vector.
	 */
	public static Vector loadVectorFromText(String inFile, JavaSparkContext sc) {
		
		// read back the file as an array of strings
		JavaRDD<String> in_read = sc.textFile(inFile);
		List<String> in_string = in_read.collect();
		
		// Since it is a vector, it has 1 dimension equal to 1
		assert(in_string.size() == 1);
		String m = in_string.get(0);
		
		// remove the square brackets
		m = m.substring(1, m.length()-1);
		
		// split the string to an array of string by using "," as a delimiter
		String[] parts = m.split(",");
		
		// convert the string to a vector
		int s = parts.length;
		double out_vector[] = new double[s];
		for (int i = 0; i < s; i++) {
			out_vector[i] = Double.parseDouble(parts[i]);
		}
		
		return Vectors.dense(out_vector);
	}
	
	
	/**
	 * Loads a Matrix from a text file. THERE IS A BUG HERE!!!!
	 * 
	 * @param inFile Input file to load.
	 * @param sc Spark context.
	 * @return Loaded matrix.
	 **/
	public static Matrix loadMatrixFromText(String inFile, JavaSparkContext sc) {
		
		// read back the file as an array of strings
		JavaRDD<String> in_read = sc.textFile(inFile);
		List<String> in_string = in_read.collect();
		
		// since it is a expanded vector, it has 1 dimension equal to 1
		int nCols = in_string.size();
		String m = in_string.get(0);
		
		// remove the square brackets
		m = m.substring(1, m.length()-1);
		
		int nRows = m.split(",").length;
		double[] out_vector = new double[nRows * nCols];
		int idx = 0;
		
		// convert the string to a matrix
		// there is a bug here !!
		for (int i = 0; i < in_string.size(); i++) {
			String m2 = in_string.get(i);
			m2 = m2.substring(1, m.length()-1);
			String[] parts = m.split(",");
			int vector_size = parts.length;
			for (int j = 0; j < vector_size; ++j) {
				out_vector[idx++] = Double.parseDouble(parts[j]);
			}
		}
		return Matrices.dense(nRows, nCols, out_vector);
	}
	
	
	/**
	 * Saves a Vector as an object.
	 * 
	 * @param input Input vector. 
	 * @param outFile Output file to save.
	 * @param sc Spark context.
	 */
	public static void saveVectorToObject(Vector input, String outFile, JavaSparkContext sc) {
		
		List<Vector> temp_input = new ArrayList<Vector>();
		temp_input.add(input);
		
		// transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsObjectFile(outFile);
	}
	
	
	/**
	 * Saves a Matrix to an object file column by column.
	 * 
	 * @param input Input matrix to save.
	 * @param outFile Output file to save.
	 * @param sc Spark context.
	 */
	//TODO:: do this row-wise
	public static void saveMatrixToObject(Matrix input, String outFile, JavaSparkContext sc) {
		
		// create a list of vectors and add one by one 
		// the columns of the matrix to the list
		List<Vector> temp_input = new ArrayList<Vector>();
		for (int i = 0; i < input.numCols(); i++) {
			double[] temp = new double[input.numRows()];
			for (int j = 0; j < input.numRows(); j++) {
				temp[j] = input.apply(j, i);
			} 
			Vector col_vector = Vectors.dense(temp);
			temp_input.add(col_vector);
		}
		
		// Transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsObjectFile(outFile);
	}
	
	
	/**
	 * Saves a Vector as a text file.
	 * 
	 * @param input Input vector.
	 * @param outFile Output file to save.
	 * @param sc Spark context.
	 */
	public static void saveVectorToText(Vector input, String outFile, JavaSparkContext sc) {
		
		List<Vector> temp_input = new ArrayList<Vector>();
		temp_input.add(input);
		
		// transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsTextFile(outFile);
	}
	
	
	/**
	 * Saves a Matrix to a text file.
	 * 
	 * @param input Input matrix.
	 * @param outFile Output file to save.
	 * @param sc Spark context.
	 */
	//TODO:: Do this row-wise.
	public static void saveMatrixToText(Matrix input, String outFile, JavaSparkContext sc) {
		
		// convert the matrix to a list of vectors. Each column is one vector. 
		List<Vector> temp_input = new ArrayList<Vector>();
		for (int i = 0; i < input.numCols(); i++) {
			double[] temp = new double[input.numRows()];
			for (int j = 0; j < input.numRows(); j++) {
				temp[j] = input.apply(j, i);
			} 
			
			// add the created column to the vector list
			Vector col_vector = Vectors.dense(temp);
			temp_input.add(col_vector);
		}
		
		// transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsTextFile(outFile);
	}
	
	
	/**
	 * Loads an array of Vectors from an object file.
	 * 
	 * @param inFile Input file to load.
	 * @param sc Spark context.
	 * @return Array of vectors.
	 */
	public static Vector[] loadVectorArrayFromObject(String inFile, JavaSparkContext sc) {
		
		// load from the object file and collect
		JavaRDD<Vector> out = sc.objectFile(inFile);
		List<Vector> a = out.collect();
		
		// convert the list of vectors to array.
		Vector[] output = new Vector[a.size()];
		for (int i = 0; i < a.size(); i++) {
			output[i] = Vectors.dense(a.get(i).toArray());
		}
		
		return output;
	}
	
	
	/**
	 * Saves an array of Vectors to an object file.
	 * 
	 * @param input Input array of vectors.
	 * @param outFile Output file to save.
	 * @param sc Spark context.
 	 */
	public static void saveVectorArrayToObject(Vector[] input, String outFile, JavaSparkContext sc) {
		
		// convert the array of vectors to a list
		List<Vector> temp_input = new ArrayList<Vector>();
		for (int i = 0; i < input.length; i++) {
			temp_input.add(input[i]);
		}
		
		// transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsObjectFile(outFile);
	}
	
	
	/**
	 * Saves an array of Vectors to a text file.
	 * 
	 * @param input Input array of vectors.
	 * @param outFile Output file to save.
	 * @param sc Spark context.
	 */
	public static void saveVectorArrayToText(Vector[] input, String outFile, JavaSparkContext sc) {
		
		// convert the array of vectors to a list 
		List<Vector> temp_input = new ArrayList<Vector>();
		for (int i = 0; i < input.length; i++) {
			temp_input.add(input[i]);
		}
		
		// transform it to JavaRDD and save it to file
		sc.parallelize(temp_input).saveAsTextFile(outFile);
	}
	
}
