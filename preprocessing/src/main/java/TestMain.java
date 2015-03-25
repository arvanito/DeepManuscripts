package main.java;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkFiles;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.input.PortableDataStream;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class TestMain {

    static final String openCVLibName = System.mapLibraryName(Core.NATIVE_LIBRARY_NAME);
    static final String openCVLibFullPath = "hdfs:///projects/deep-learning/lib/"+ openCVLibName;

    /**
     * Load OpenCV library, first try in local directory, if it does not work, switch to the file uploaded by the driver
     */
    static public void loadLibrary() {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            System.out.println("OpenCV found in java.library.path");
        }catch (UnsatisfiedLinkError e) {
            System.out.println("OpenCV not found in java.library.path");
            System.out.print("Trying SparkFiles...");
            try {
                System.load(SparkFiles.get(openCVLibName));
                System.out.println("SUCCESS");
            }catch (Exception e2) {
                System.out.println("FAILURE");
            }
        }
    }

    //For worker nodes, will probably fail on driver node
    //static {
    //    loadLibrary();
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
            sc.addFile(openCVLibFullPath);
            inputFile = args[0];
            outputFile = args[1];
        }

        //Get a handle for every file in the directory
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(inputFile);

        //Stupid (until I find better) way to load the library, apply an identity mapPartitions thingy that loads it
        loadLibrary();
        dataStream = dataStream.mapPartitionsToPair(new PairFlatMapFunction<Iterator<Tuple2<String,PortableDataStream>>, String, PortableDataStream>() {
            @Override
            public Iterable<Tuple2<String, PortableDataStream>> call(Iterator<Tuple2<String, PortableDataStream>> input) throws Exception {
                loadLibrary();
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
            } });

        //Get the number of pixels for each image
        JavaPairRDD<String, Integer> pixelsNumber = dataImages.mapValues(new Function<ImageData, Integer>() {
            public Integer call(ImageData im) {
                Mat m = im.getImage(); //Decompress and return a pointer to the uncompressed image representation
                    return m.cols()*m.rows();
            } } );

        //Save result
        pixelsNumber.saveAsTextFile(outputFile);
        sc.close();
    }
}