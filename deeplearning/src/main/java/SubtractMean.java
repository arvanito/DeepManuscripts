package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Subtract mean vector from a vector in parallel.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class SubtractMean implements Function<Vector, Vector> {

	private static final long serialVersionUID = 3678255317238045810L;
	private Vector mean;	// mean Vector
	
	
	/**
	 * Constructor
	 * 
	 * @param mean Input mean Vector
	 */
	SubtractMean(Vector mean) {
		this.mean = mean;
	}
	
	
	/**
	 * Method that is called during a map call.
	 * 
	 * @param v Input Vector
	 * @param m Mean Vector
	 * @return Subtracted Vector result
	 */
	@Override
	public Vector call(Vector v) {		
		
		if (v.size() != mean.size()) {
			throw new IllegalArgumentException("Vector sizes are incompatible!");
		}
		
		// Vector size
		int s = v.size();

		// loop over elements to subtract the two Vectors
		double[] sub = new double[s];
		for (int i = 0; i < s; i++) {
			sub[i] = v.apply(i) - mean.apply(i);
		}
		
		return Vectors.dense(sub);
	}
	
}
