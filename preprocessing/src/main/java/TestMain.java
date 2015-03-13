package main.java;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.linalg.Vector;
import org.opencv.core.Core;
import org.opencv.core.Mat;

public class TestMain {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("DeepManuscript preprocessing").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        String inputFile = args[0];
        String outputFile = args[1];
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(inputFile);
        JavaPairRDD<String,ImageData> dataImages = dataStream.mapValues(new ImageDataCreator());

        JavaPairRDD<String,Integer> pixelsNumber = dataImages.mapValues( (ImageData im) -> {Mat m =im.getImage(); return m.cols()*m.rows(); } );

        pixelsNumber.saveAsTextFile(outputFile);
        sc.close();
    }
}