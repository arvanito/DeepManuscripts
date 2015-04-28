package main.java;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.opencv.core.Mat;
import ch.epfl.dhlab.AndreaPipeline;
import org.opencv.highgui.Highgui;
import scala.Tuple2;

public class AndreaPipelineMain {

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

        //Get a handle for every file in the directory
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(inputFile);


        //Convert the PortableDataStream to ImageData representations
        JavaPairRDD<String,ImageData> dataImages = dataStream.mapValues(new Function<PortableDataStream, ImageData>() {
            public ImageData call(PortableDataStream portableDataStream) throws Exception {
                return new ImageData(portableDataStream.toArray());
            }
        });

        //Segment the image, results as : <fileBasename,<jSonData,imgSegmentation>>
        JavaRDD<Tuple2<String, Tuple2<String, ImageData>>> segmentationResult = dataImages.map(new Function<Tuple2<String, ImageData>, Tuple2<String, Tuple2<String, ImageData>>>() {
            public Tuple2<String, Tuple2<String, ImageData>> call(Tuple2<String, ImageData> data) {
                Mat m = data._2().getImage(); //Decompress and return a pointer to the uncompressed image representation
                Mat binarized = AndreaPipeline.binarizePage(m); // Binarize the image
                Mat segmentationResult = new Mat();
                String jSonString = AndreaPipeline.lineDetection(data._1(), m, binarized, segmentationResult); //detect the lines
                String basename = FilenameUtils.getBaseName(data._1());
                return new Tuple2<String, Tuple2<String, ImageData>>(basename, new Tuple2<String, ImageData>(jSonString, new ImageData(segmentationResult)));
            }
        });

        //Filtering only the jSon output
        JavaPairRDD<Text, Text> jSonData = JavaPairRDD.fromJavaRDD(segmentationResult.map(new Function<Tuple2<String, Tuple2<String, ImageData>>, Tuple2<Text, Text>>() {
            public Tuple2<Text, Text> call(Tuple2<String, Tuple2<String, ImageData>> data) {
                String basename = data._1();
                return new Tuple2<Text, Text>(new Text(basename+".json"),new Text(data._2()._1()));
            }
        }));

        //Save jSon files separately based on the key value
        jSonData.saveAsHadoopFile(outputFile,Text.class,Text.class,MultiFileOutput.class);

        sc.close();
    }
}

