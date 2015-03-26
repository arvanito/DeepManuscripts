package main.java;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Main processing class, input folder contains images, output folder contains the randomly extracted patches
 */
public class PreprocessMain {

    //Ultimately a static declaration like that should do the job
    //static {
    //    OpenCV.loadLibrary();
    //}

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
            //Send OpenCV native lib to all worker
            sc.addFile(OpenCV.openCVLibFullPath);
            inputFile = args[0];
            outputFile = args[1];
        }

        //Get a handle for each file in the input directory
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(inputFile);

        //Stupid (until I find better) way to load the library, apply an identity mapPartitions thingy that loads it
        OpenCV.loadLibrary();
        dataStream = dataStream.mapPartitionsToPair(new PairFlatMapFunction<Iterator<Tuple2<String,PortableDataStream>>, String, PortableDataStream>() {
            @Override
            public Iterable<Tuple2<String, PortableDataStream>> call(Iterator<Tuple2<String, PortableDataStream>> input) throws Exception {
                OpenCV.loadLibrary();
                List<Tuple2<String, PortableDataStream> > l = new ArrayList<>();
                while(input.hasNext())
                    l.add(input.next());
                return l;
            }
        });

        //Convert the PortableDataStream to ImageData representations
        JavaPairRDD<String,ImageData> dataImages = dataStream.mapValues( new Function<PortableDataStream, ImageData>() {
            public ImageData call(PortableDataStream portableDataStream) throws Exception {
                return new ImageData(portableDataStream.toArray());
            } } );

        //Extract lines
        //TODO

        //Extract patches
        JavaPairRDD<String, Vector> patches = dataImages.flatMapValues(new Function<ImageData, Iterable<Vector> >() {
            public Iterable<Vector> call(ImageData im) {
                return ImageFunctions.extractPatches(im.getImage(), 10);
            }
        });

        //Save result
        patches.values().saveAsTextFile(outputFile);
        sc.close();
    }
}