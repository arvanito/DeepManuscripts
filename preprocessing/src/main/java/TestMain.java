package main.java;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.linalg.Vector;
import org.opencv.core.Core;

public class TestMain {

    //Load OpenCV JNI
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        String inputFile, outputFile;
        SparkConf conf;
        JavaSparkContext sc;
        if(args[0].equals("--local")) {
            conf = new SparkConf().setAppName("DeepManuscript preprocessing").setMaster("local");
            sc = new JavaSparkContext(conf);
            inputFile = args[1];
            outputFile = args[2];
        }else {
            conf = new SparkConf().setAppName("DeepManuscript preprocessing");
            sc = new JavaSparkContext(conf);
            inputFile = args[0];
            outputFile = args[1];
        }

        //Get a handle for each file in the input directory
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(inputFile);

        //Convert the PortableDataStream to ImageData representations
        JavaPairRDD<String,ImageData> dataImages = dataStream.mapValues( new Function<PortableDataStream, ImageData>() {
            public ImageData call(PortableDataStream portableDataStream) throws Exception {
                return new ImageData(portableDataStream.toArray());
            } } );

        //Extract patches
        JavaPairRDD<String, Vector> patches = dataImages.flatMapValues(new Function<ImageData, Iterable<Vector> >() {
            public Iterable<Vector> call(ImageData im) {
                return ImageFunctions.extractPatches(im.getImage(), 10);
            }
        });

        //Get the number of pixels for each image
/*        JavaPairRDD<String,Integer> pixelsNumber = dataImages.mapValues( new Function<ImageData, Integer>() {
            public Integer call(ImageData im) {
                    Mat m =im.getImage(); //Decompress and return a pointer to the uncompressed image representation
                    return m.cols()*m.rows();
            } } );*/

        //Save result
        patches.saveAsTextFile(outputFile);
        sc.close();
    }
}