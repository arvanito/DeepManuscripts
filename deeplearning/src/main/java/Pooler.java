package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;

/**
 * All Poolers (Max pool, Group-and-maxpool,...) should implement this interface. See DummyPooler for example.
 * 
 * @author Arttu Voutilainen
 *
 */
public interface Pooler extends Function<Vector,Vector>{

}
