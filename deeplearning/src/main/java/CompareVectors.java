package main.java;

import java.io.Serializable;
import java.util.Comparator;

import org.apache.spark.mllib.linalg.Vector;

/**
 * Helper class that implements the Comparator interface to compare 
 * two Vectors numerically.
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class CompareVectors implements Comparator<Vector>, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 299004719346431001L;

	/**
	 * Overloaded compare from the Comparator interface. Two Vectors are compared 
	 * value to value, using the Double.compare().
	 * 
	 * @param v1 First input Vector
	 * @param v2 Second input Vector
	 * @return Output of Double.compare()
	 * @throws IllegalArgumentException
	 */
	@Override
	public int compare(Vector v1, Vector v2) throws IllegalArgumentException  {

		int s1 = v1.size();
		int s2 = v2.size();
		if (s1 != s2) {
			throw new IllegalArgumentException("The two Vectors do not have the same length!");
		}
		
		// compare the Vectors numerically (lexicographical order)
		int compValue;
		for (int i = 0; i < s1; i++) {
			
			// use of Double.compare from Java
			compValue = Double.compare(v1.apply(i), v2.apply(i));

			if (compValue != 0) {
				return compValue;
			}
		}
		return 0;
	}
	
}
