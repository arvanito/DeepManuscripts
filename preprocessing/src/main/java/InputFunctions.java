package main.java;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import scala.Tuple2;

import java.io.IOException;

/**
 * Helper class to load files from directories
 */
public class InputFunctions {

    /**
     * Load images from directories
     * @param sc the current spark context
     * @param inputFolder where to load the data from
     * @param regex specifies sub-directories/elements from inputFolder (only wildcards allowed)
     * @return PairRDD of (filename : String, img : ImageData) where filename is local from inputFolder
     */
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

        final String fInputFileAbs = findAbsolutePath(sc,inputFolder);
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


    private static String findAbsolutePath(JavaSparkContext sc, String folder) {
        //Find absolute path of input folder
        String inputFileAbs="";
        FileSystem fs=null;
        try {
            fs=FileSystem.get(sc.hadoopConfiguration());
            inputFileAbs = fs.resolvePath(new Path(folder)).toString();
            System.out.println("Input path : " + inputFileAbs);
        } catch (IOException e) {
            e.printStackTrace();
            new Error("Unable to locate input path");
        }
        return inputFileAbs;
    }

}
