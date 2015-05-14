package test.java;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.List;

import main.java.KNearestNeighbor;
import main.java.SpectralClustering;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class SpectralClusteringTest {

	public static void main(String[] args) {
		PrintWriter writer = null;
		try {
			writer = new PrintWriter("Test1.txt", "UTF-8");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double testArray[][] = {{1,2,3,4,5,6},{1,2,3,4,5,7},{2,4,6,8,10,12},{2,3,4,5,6,7},{5,5,5,5,5,5}};
		Vector[] vectors = new Vector[5];

		for (int i = 0; i < 5; i++) {
			vectors[i] = Vectors.dense(testArray[i]);
			writer.println("Vector " + i +": " + vectors[i].toArray()[0] + " " + vectors[i].toArray()[1] + " " + vectors[i].toArray()[2] + " " + vectors[i].toArray()[3] + " " + vectors[i].toArray()[4] + " " + vectors[i].toArray()[5] + " ");
		}
		
		SpectralClustering sC = new SpectralClustering(vectors);
		sC.computeKNN(3, 1, 1, 0.5);
		sC.computeClustering(2);
		JavaRDD<Integer> result = sC.getVectorsClusters();
		List<Integer> resultList = result.toArray();
		Integer[] resultArray = null;
		resultArray = resultList.toArray(resultArray);
		writer.println("Matrix:");
		for(int i=0; i<5; i++){
				writer.print(resultArray[i].toString() + " ");
				writer.print("\n");
		}
		writer.println("The first line");
		writer.println("The second line");
		writer.close();

	}
	


}
