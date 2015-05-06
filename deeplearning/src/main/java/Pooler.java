package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

/**
 * All Poolers (Max pool, Group-and-maxpool,...) should implement this interface. See DummyPooler for example.
 * 
 * @author Arttu Voutilainen
 *
 */
public interface Pooler extends Function<Tuple2<Vector, Vector>, Tuple2<Vector, Vector>>{

	/**
	 * Main method that implements pooling and non-linear activation.
	 * This method will be passed to a map call and applied independently to each data point.
	 * 
	 * @param data A Vector representing one data point
	 * @return A Vector after applying pooling and non-linear activation
	**/
	public Tuple2<Vector, Vector> call(Tuple2<Vector, Vector> data) throws Exception;
}
