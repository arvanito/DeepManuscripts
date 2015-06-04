package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigKMeans;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 * K-Means learning, using the default K-Means from Spark.
 * 
 * @author Arttu Voutilainen
 *
 */
public class KMeansLearner implements Learner {

	private static final long serialVersionUID = -3835838697603177208L;

	final int numClusters;
	final int numIterations;
	final String initType;
	
	
	/**
	 * Constructor that set the current base layer configuration.
	 * Sets the number of cluster, number of iterations and type of initialization.
	 * 
	 * @param configLayer Current base layer configuration object.
	 */
	public KMeansLearner(ConfigBaseLayer configLayer) {
		ConfigKMeans conf = configLayer.getConfigKmeans();
		numClusters = conf.getNumberOfClusters();
		numIterations = conf.getNumberOfIterations();
		if (conf.getType() == 1) {
			initType = KMeans.RANDOM();
		} else {
			initType = KMeans.K_MEANS_PARALLEL();
		}
	}
	
	
	/**
	 * Method that performs the feature learning.
	 * 
	 * @param pairData Input dataset from which we learn the features.
	 * @return Array of learned features.
	 */
	@Override
	public Vector[] call(JavaRDD<Tuple2<Vector, Vector>> pairData) {
		
		// extract second part of the pair
		JavaRDD<Vector> data = pairData.map(
					new Function<Tuple2<Vector, Vector>, Vector>() {
						private static final long serialVersionUID = 6369401581724529416L;

						public Vector call(Tuple2<Vector, Vector> pair) {
							return pair._2;
						}
					}
				);
		
		// run K-means
	    KMeansModel clusters = KMeans.train(data.rdd(), numClusters, numIterations, 5, initType);
	    
	    // return the cluster centers as the learned features
		return clusters.clusterCenters();
	}
	
	
	/**
	 * Method that loads the learned features from a file.
	 * 
	 * @param filename Input file name to load.
	 * @param sc Spark context.
	 * @return Array of learned features.
	 */
	@Override
	public Vector[] loadFromFile(String filename, JavaSparkContext sc) {
		return LinAlgebraIOUtils.loadVectorArrayFromObject(filename, sc);
	}
	
	
	/**
	 * Method that saves the learned features to an object file.
	 * 
	 * @param features Input array of vectors to save.
	 * @param filename Input file name to save.
	 * @param sc Spark context.
	 */
	@Override
	public void saveToFile(Vector[] features, String filename, JavaSparkContext sc) {
		LinAlgebraIOUtils.saveVectorArrayToObject(features, filename, sc);
	}

}
