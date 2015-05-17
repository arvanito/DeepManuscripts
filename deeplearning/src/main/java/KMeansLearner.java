package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigKMeans;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;

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
	
	public KMeansLearner(ConfigBaseLayer configLayer) {
		ConfigKMeans conf = configLayer.getConfigKmeans();
		numClusters = conf.getNumberOfClusters();
		numIterations = conf.getNumberOfIterations();
	}
	
	@Override
	public Vector[] call(JavaRDD<Vector> data) throws Exception {
	    KMeansModel clusters = KMeans.train(data.rdd(), numClusters, numIterations, 5, KMeans.K_MEANS_PARALLEL());
		return clusters.clusterCenters();
	}

}
