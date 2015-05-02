package main.java;

import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigKMeans;

import org.apache.spark.api.java.JavaRDD;
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
	
	public KMeansLearner(ConfigBaseLayer configLayer) {
		ConfigKMeans conf = configLayer.getConfigKmeans();
		numClusters = conf.getNumberOfClusters();
		numIterations = conf.getNumberOfIterations();
	}
	
	@Override
	public Vector[] call(JavaRDD<Tuple2<Vector, Vector>> pair_data) throws Exception {
		JavaRDD<Vector> data = pair_data.map(
				new Function<Tuple2<Vector, Vector>, Vector>() {
					private static final long serialVersionUID = 6369401581724529416L;

					public Vector call(Tuple2<Vector, Vector> pair) {
						return pair._2;
					}
				} 
		);
	    KMeansModel clusters = KMeans.train(data.rdd(), numClusters, numIterations);
		return clusters.clusterCenters();
	}

}
