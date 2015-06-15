package ch.epfl.ivrl.deepmanuscripts;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 * Class that performs the computation of similarities between vector representations.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class ComputeSimilarityPair implements Function<Tuple2<Vector, Vector>, Tuple2<Vector, Double>> {

	private static final long serialVersionUID = 7260655554465270778L;
	
	private Tuple2<Vector, Vector> query;
	private double normQuery; 
	
	
	/**
	 * Constructor. It sets the query representation.
	 * 
	 * @param query The query representation.
	 */
	public ComputeSimilarityPair(Tuple2<Vector, Vector> query) {
		this.query = query;
		NormQuery();
	}
	
	
	/**
	 * Method that computes the norm of the query.
	 */
	public void NormQuery() {
		normQuery = Math.sqrt(BLAS.dot(query._2, query._2));
	}
	
	
	/**
	 * Method that is called during a map call. It computes the 
	 * cosine similarity between the query representation and 
	 * the representations of the candidate image patches.
	 * 
	 * @param v Candidate patch representation.
	 * @return Cosine similarity between query and current patch.
	 */
	public Tuple2<Vector, Double> call(Tuple2<Vector, Vector> v) {
		
		// compute the norm of the input vector
		double normV = Math.sqrt(BLAS.dot(v._2, v._2));
		
		// compute cosine similarity between the query and the input vector
		double dotP = BLAS.dot(query._2, v._2) / (normQuery * normV);
		
		return new Tuple2<Vector, Double>(v._1, dotP);
	}

}
