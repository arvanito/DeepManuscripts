package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;

/**
 * All Extractors (Matrix multiply, Convolution,..) should implement this interface. See DummyExtractor for example.
 * 
 * @author Arttu Voutilainen
 *
 */
public interface Extractor extends Function<Vector, Vector> {

}
