package main.java;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.storage.StorageLevel;
import org.opencv.core.Mat;
import ch.epfl.dhlab.AndreaPipeline;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import scala.Tuple2;

import java.io.IOException;

public class AndreaPipelineMain {

    public static JavaPairRDD<String,ImageData> loadImages(JavaSparkContext sc, String inputFolder, String regex) {
        //Get a handle for every file in the directory
        JavaPairRDD<String,PortableDataStream> dataStream  = sc.binaryFiles(FilenameUtils.concat(inputFolder, regex))
        //Filter non-image file
            .filter(new Function<Tuple2<String, PortableDataStream>, Boolean>() {
                @Override
                public Boolean call(Tuple2<String, PortableDataStream> d) throws Exception {
                    String ext = FilenameUtils.getExtension(d._1());
                    return ext.equals("jpg") || ext.equals("JPG") || ext.equals("jpeg") || ext.equals("JPEG")
                            || ext.equals("png") || ext.equals("PNG") || ext.equals("tif") || ext.equals("TIF")
                            || ext.equals("tiff") || ext.equals("TIFF");
                }
            });
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

        JavaPairRDD<String, ImageData> dataImages = loadImages(sc,inputFile,dirRegex);

        //Segment the image, results as : <fileBasename,<jSonData,imgSegmentation>>
        JavaPairRDD<String, Tuple2<String, ImageData>> segmentationResult = dataImages.mapToPair(new PairFunction<Tuple2<String, ImageData>, String, Tuple2<String, ImageData>>() {
            public Tuple2<String, Tuple2<String, ImageData>> call(Tuple2<String, ImageData> data) {
                Mat m = data._2().getImage(); //Decompress and return a pointer to the uncompressed image representation
                Mat segmentationResult = new Mat();
                String jSonStringResult = "";
                if(m.cols()>0 && m.rows()>0) {
                    Mat m2 = new Mat();
                    Imgproc.resize(m, m2, new Size(m.cols()/2,m.rows()/2));
                    Mat binarized = AndreaPipeline.binarizePage(m2); // Binarize the image
                    if (binarized.size().equals(m2.size())) {
                        Imgproc.resize(binarized, binarized, m.size());
                        String jSonString = AndreaPipeline.lineDetection(data._1(), m, binarized, segmentationResult); //detect the lines
                        if (segmentationResult.size().equals(m.size())) {
                            int newHeight = 1200;
                            int newWidth = newHeight * segmentationResult.cols() / segmentationResult.rows();
                            Imgproc.resize(segmentationResult, segmentationResult, new Size(newWidth, newHeight));
                            jSonStringResult = jSonString;
                        }
                    }
                }
                return new Tuple2<String, Tuple2<String, ImageData>>(data._1(), new Tuple2<String, ImageData>(jSonStringResult, new ImageData(segmentationResult)));
            }
        });
        //Persist result so it is not recomputed multiple times
        segmentationResult.cache();

        //Save failed elements
        JavaPairRDD<String, String> segmentationResultFailed = segmentationResult.filter(new Function<Tuple2<String, Tuple2<String, ImageData>>, Boolean>() {
            @Override
            public Boolean call(Tuple2<String, Tuple2<String, ImageData>> stringTuple2Tuple2) throws Exception {
                return stringTuple2Tuple2._2()._1() == "";
            }
        }).mapToPair(new PairFunction<Tuple2<String, Tuple2<String, ImageData>>, String, String>() {
            public Tuple2<String, String> call(Tuple2<String, Tuple2<String, ImageData>> data) {
                return new Tuple2<String, String>(data._1(), data._2()._1());
            }
        });
        //Save empty files with the names of failed elements
        segmentationResultFailed.saveAsHadoopFile(outputFile+"-failed", String.class, String.class, MultipleStringFileOutputFormat.class);


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
        jSonData.saveAsHadoopFile(outputFile, String.class, String.class, MultipleStringFileOutputFormat.class);

        //Filtering only the marked segmentation
        JavaPairRDD<String, ImageData> imgOutput = segmentationResultSuccess.mapToPair(new PairFunction<Tuple2<String, Tuple2<String, ImageData>>, String, ImageData>() {
            public Tuple2<String, ImageData> call(Tuple2<String, Tuple2<String, ImageData>> data) {
                String basename = data._1();
                return new Tuple2<String, ImageData>(basename, data._2()._2());
            }
        });
        //Save segmentation representation
        imgOutput.saveAsHadoopFile(outputFile+"-img", String.class, ImageData.class, ImageOutputFormat.class);

        sc.close();
    }
}

