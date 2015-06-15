package ch.epfl.ivrl.deepmanuscripts;

import java.io.Serializable;
import java.util.Comparator;

import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

public class VectorComparator implements Comparator<Tuple2<Vector,Double>>, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1103000639693317307L;

	@Override
	public int compare(Tuple2<Vector, Double> o1, Tuple2<Vector, Double> o2) {
		return o1._2 < o2._2 ? -1 : o1._2 == o2._2 ? 0 : 1;
	}

}
