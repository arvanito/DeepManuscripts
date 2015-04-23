package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;

/**
 * Class that performs the computation of similarities between vector representations.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class ComputeSimilarity implements Function<Vector, Double> {
	
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 7260655554465270778L;
	private Vector query;	// query representation
	private double normQuery; //
	
	
	/**
	 * Constructor. It sets the query representation.
	 * 
	 * @param query The query representation
	 */
	public ComputeSimilarity(Vector query) {
		this.query = query;
		NormQuery();
	}
	
	
	/**
	 * Method that computes the norm of the query
	 */
	public void NormQuery() {
		normQuery = Math.sqrt(BLAS.dot(query, query));
	}
	
	
	/**
	 * Method that is called during a map call. It computes the 
	 * cosine similarity between the query representation and 
	 * the representations of the candidate image patches
	 * 
	 * @param v Candidate patch representation
	 * @return Cosine similarity between query and current patch
	 */
	public Double call(Vector v) {
		
		// compute the norm of the input vector
		double normV = Math.sqrt(BLAS.dot(v, v));
		
		// compute cosine similarity between the query and the input vector
		double dotP = BLAS.dot(query, v) / (normQuery * normV);
		
		return dotP;
	}

}
