package main.java;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.linalg.Vector;

/**
 * Main processing class, input folder contains images, output folder contains the randomly extracted patches
 */
public class PreprocessMain {

    public static void main(String[] args) {
        String inputFile, outputFile;
        int nbPatches;
        SparkConf conf;
        JavaSparkContext sc;
        if(args[0].equals("--local")) {
            conf = new SparkConf().setAppName("DeepManuscript preprocessing").setMaster("local");
            sc = new JavaSparkContext(conf);
            inputFile = args[1];
            outputFile = args[2];
            nbPatches = Integer.valueOf(args[3]);
        }else {
            conf = new SparkConf().setAppName("DeepManuscript preprocessing");
            sc = new JavaSparkContext(conf);
            inputFile = args[0];
            outputFile = args[1];
            nbPatches = Integer.valueOf(args[2]);
        }

        //Get a handle for each file in the input directory
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(inputFile);

        //Convert the PortableDataStream to ImageData representations
        JavaPairRDD<String,ImageData> dataImages = dataStream.mapValues( new Function<PortableDataStream, ImageData>() {
            public ImageData call(PortableDataStream portableDataStream) throws Exception {
                return new ImageData(portableDataStream.toArray());
            } } );

        //Extract lines
        //TODO

        //Extract patches
        final int nbPatchesPerImg = (int)(nbPatches/dataImages.count());
        JavaPairRDD<String, Vector> patches = dataImages.flatMapValues(new Function<ImageData, Iterable<Vector> >() {
            public Iterable<Vector> call(ImageData im) {
                return ImageFunctions.extractPatches(im.getImage(), nbPatchesPerImg);
            }
        });

        //Save result
        patches.values().saveAsTextFile(outputFile);
        sc.close();
    }
}