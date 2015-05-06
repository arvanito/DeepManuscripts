package main.java;


import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import scala.Tuple2;

import java.io.IOException;

public class TestMain {

    public static JavaPairRDD<String,ImageData> loadImages(JavaSparkContext sc, String inputFolder, String regex) {
        //Get a handle for every file in the directory
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(FilenameUtils.concat(inputFolder, regex));
        //Find absolute path of input folder
        String inputFileAbs="";
        FileSystem fs=null;
        try {
            fs=FileSystem.get(sc.hadoopConfiguration());
            inputFileAbs = fs.resolvePath(new Path(inputFolder)).toString();
            System.out.println("Input path : " + inputFileAbs);
        } catch (IOException e) {
            e.printStackTrace();
            new Error("Unable to locate input path");
        }
        final String fInputFileAbs = inputFileAbs;
        final FileSystem ffs = fs;
        //Remove the input path and extract ImageData from the PortableDataStream
        JavaPairRDD<String, ImageData> dataImages = dataStream.mapToPair(new PairFunction<Tuple2<String, PortableDataStream>, String, ImageData>() {
            public Tuple2<String, ImageData> call(Tuple2<String, PortableDataStream> data) {
                String filename = data._1().substring(fInputFileAbs.length() + 1);
                //String basename = FilenameUtils.removeExtension(filename);
                return new Tuple2<String, ImageData>(filename, new ImageData(data._2().toArray()));
            }
        });
        return dataImages;
    }



    public static void main(String[] args) {
        final String inputFile, outputFile;
        String dirRegex = "*.tif";
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

        JavaPairRDD<String, ImageData> dataImages = loadImages(sc,inputFile,dirRegex);

        //Get the number of pixels for each image
        JavaPairRDD<String, String> pixelsNumber = dataImages.mapValues(new Function<ImageData, String>() {
            public String call(ImageData data) {
                Mat img = data.getImage();
                return Integer.toString(img.cols() * img.rows());
            }
        });


        //Save result
        pixelsNumber.saveAsHadoopFile(outputFile, String.class, String.class, MultiFileOutput.class);

        //Save result
       // dataImages.saveAsHadoopFile(outputFile+"-img", String.class, ImageData.class, ImageOutputFormat.class);

        sc.close();
    }
}