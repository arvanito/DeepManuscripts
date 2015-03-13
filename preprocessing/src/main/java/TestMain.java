package main.java;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.linalg.Vector;
import org.opencv.core.Core;
import org.opencv.core.Mat;

public class TestMain {

    //Load OpenCV JNI
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("DeepManuscript preprocessing").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        String inputFile = args[0];
        String outputFile = args[1];

        //Get a handle for each file in the input directory
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(inputFile);

        //Convert the PortableDataStream to ImageData representations
        JavaPairRDD<String,ImageData> dataImages = dataStream.mapValues( new Function<PortableDataStream, ImageData>() {
            public ImageData call(PortableDataStream portableDataStream) throws Exception {
                return new ImageData(portableDataStream.toArray());
            } } );

        //Get the number of pixels for each image
        JavaPairRDD<String,Integer> pixelsNumber = dataImages.mapValues( new Function<ImageData, Integer>() {
            public Integer call(ImageData im) {
                    Mat m =im.getImage(); //Decompress and return a pointer to the uncompressed image representation
                    return m.cols()*m.rows();
            } } );

        //Save result
        pixelsNumber.saveAsTextFile(outputFile);
        sc.close();
    }
}