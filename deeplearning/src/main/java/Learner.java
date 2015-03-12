package main.java;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;

/**
 * All Learners (KMeans, Autoencoders,..) should implement this interface. See DummyLearner for example.
 * 
 * @author Arttu Voutilainen
 *
 */
public interface Learner extends Function<JavaRDD<Vector>,Vector[]>{

}
