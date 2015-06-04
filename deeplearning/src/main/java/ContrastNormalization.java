package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;


/**
 * Contrast Normalization of a Vector point in parallel.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class ContrastNormalization implements Function<Vector, Vector> {

	private static final long serialVersionUID = 610489156631574927L;
	private double e;	// epsilon parameter
	
	
	/**
	 * Constructor.
	 * 
	 * @param e Parameter for contrast normalization.
	 */
	public ContrastNormalization(double e) {
		this.e = e;
	}


	/**
	 * Method that is called during a map call.
	 * 
	 * @param v Input Vector.
	 * @return The processed contrast normalized Vector.
	 */
	@Override
	public Vector call(Vector v) {	

		// compute mean value of the Vector
		int s = v.size();
		double m = 0;
		for (int i = 0; i < s; i++) {
			m += v.apply(i);
		}
		m /= s;

		// compute standard deviation of the Vector
		double stdev = 0;
		for (int i = 0; i < s; i++) {
			stdev += (v.apply(i) - m) * (v.apply(i) - m);
		}
		stdev = stdev / (s - 1);

		// subtract mean and divide by the standard deviation
		for (int i = 0; i < s; i++) {
			v.toArray()[i] = v.apply(i) - m;
			v.toArray()[i] = v.apply(i) / Math.sqrt((stdev + e));
		}

		return v;
	}
	
}
