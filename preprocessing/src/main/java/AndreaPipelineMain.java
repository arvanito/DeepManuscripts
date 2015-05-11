package main.java;

import ch.epfl.dhlab.AndreaPipeline;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import scala.Tuple2;

import java.util.LinkedList;
import java.util.List;

public class AndreaPipelineMain {


    public static void main(String[] args) {
        String inputFile, outputFile, dirRegex;
        SparkConf conf;
        JavaSparkContext sc;
        if(args[0].equals("--local")) {
            conf = new SparkConf().setAppName("DeepManuscript preprocessing").setMaster("local");
            sc = new JavaSparkContext(conf);
            inputFile = args[1];
            dirRegex = args[2];
            outputFile = args[3];
        }else {
            conf = new SparkConf().setAppName("DeepManuscript preprocessing");
            sc = new JavaSparkContext(conf);
            inputFile = args[0];
            dirRegex = args[1];
            outputFile = args[2];
        }

        JavaPairRDD<String, ImageData> dataImages = InputFunctions.loadImages(sc, inputFile, dirRegex);

        //Crop the images
        JavaPairRDD<String, ImageData> pagesCropped = dataImages.flatMapValues(new Function<ImageData, Iterable<ImageData>>() {
            public Iterable<ImageData> call(ImageData data) {
                Mat m = data.getImage(); //Decompress and return a pointer to the uncompressed image representation
                List<ImageData> result = new LinkedList<ImageData>();
                if (m.cols() > 0 && m.rows() > 0)
                    result.add( new ImageData(AndreaPipeline.cropPage(m)) );
                return result;
            }
        });
        //Save cropped images
        pagesCropped.saveAsHadoopFile(outputFile+"-cropped", String.class, ImageData.class, ImageOutputFormat.class);

        //Segment the image, results as : <fileBasename,<jSonData,imgSegmentation>>
        JavaPairRDD<String, Tuple2<String, ImageData>> segmentationResult = pagesCropped.mapToPair(new PairFunction<Tuple2<String, ImageData>, String, Tuple2<String, ImageData>>() {
            public Tuple2<String, Tuple2<String, ImageData>> call(Tuple2<String, ImageData> data) {
                Mat m = data._2().getImage(); //Decompress and return a pointer to the uncompressed image representation
                Mat segmentationResult = m;
                String jSonStringResult = "";
                if(m.cols()>0 && m.rows()>0) {
                    Mat m2 = new Mat();
                    Imgproc.resize(m, m2, new Size(m.cols() / 2, m.rows() / 2));
                    Mat binarized = AndreaPipeline.binarizePage(m2); // Binarize the image
                    if (binarized.size().equals(m2.size()) && Core.countNonZero(binarized)>0.7*binarized.cols()*binarized.rows()) {
                        Imgproc.resize(binarized, binarized, m.size());
                        Mat segmentation = new Mat();
                        String jSonString = AndreaPipeline.lineDetection(data._1(), m, binarized, segmentation); //detect the lines
                        if (segmentation.size().equals(m.size()) && !jSonString.equals("")) {
                            int newHeight = 1200;
                            int newWidth = newHeight * segmentation.cols() / segmentation.rows();
                            Imgproc.resize(segmentation, segmentation, new Size(newWidth, newHeight));
                            //assign results
                            jSonStringResult = jSonString;
                            segmentationResult = segmentation;
                        }
                    }
                }
                return new Tuple2<String, Tuple2<String, ImageData>>(data._1(), new Tuple2<String, ImageData>(jSonStringResult, new ImageData(segmentationResult)));
            }
        });
        //Persist result so it is not recomputed multiple times
        segmentationResult.cache();

        //Save failed elements
        JavaPairRDD<String, ImageData> segmentationResultFailed = segmentationResult.filter(new Function<Tuple2<String, Tuple2<String, ImageData>>, Boolean>() {
            @Override
            public Boolean call(Tuple2<String, Tuple2<String, ImageData>> stringTuple2Tuple2) throws Exception {
                return stringTuple2Tuple2._2()._1() == "";
            }
        }).mapToPair(new PairFunction<Tuple2<String, Tuple2<String, ImageData>>, String, ImageData>() {
            public Tuple2<String, ImageData> call(Tuple2<String, Tuple2<String, ImageData>> data) {
                return new Tuple2<String, ImageData>(data._1(), data._2()._2());
            }
        });
        //Save empty files with the names of failed elements
        segmentationResultFailed.saveAsHadoopFile(outputFile+"-failed", String.class, ImageData.class, ImageOutputFormat.class);


        //Save successfull elements
        JavaPairRDD<String, Tuple2<String, ImageData>> segmentationResultSuccess = segmentationResult.filter(new Function<Tuple2<String, Tuple2<String, ImageData>>, Boolean>() {
            @Override
            public Boolean call(Tuple2<String, Tuple2<String, ImageData>> stringTuple2Tuple2) throws Exception {
                return stringTuple2Tuple2._2()._1() != "";
            }
        });

        //Filtering only the jSon output
        JavaPairRDD<String, String> jSonData = segmentationResultSuccess.mapToPair(new PairFunction<Tuple2<String, Tuple2<String, ImageData>>, String, String>() {
            public Tuple2<String, String> call(Tuple2<String, Tuple2<String, ImageData>> data) {
                String basename = FilenameUtils.removeExtension(data._1());
                return new Tuple2<String, String>(basename + ".json", data._2()._1());
            }
        });
        //Save jSon files separately based on the key value
        jSonData.saveAsHadoopFile(outputFile+"-json", String.class, String.class, MultipleStringFileOutputFormat.class);

        //Filtering only the marked segmentation
        JavaPairRDD<String, ImageData> imgOutput = segmentationResultSuccess.mapToPair(new PairFunction<Tuple2<String, Tuple2<String, ImageData>>, String, ImageData>() {
            public Tuple2<String, ImageData> call(Tuple2<String, Tuple2<String, ImageData>> data) {
                String basename = data._1();
                return new Tuple2<String, ImageData>(basename, data._2()._2());
            }
        });
        //Save segmentation representation
        imgOutput.saveAsHadoopFile(outputFile+"-seg", String.class, ImageData.class, ImageOutputFormat.class);

        sc.close();
    }
}

