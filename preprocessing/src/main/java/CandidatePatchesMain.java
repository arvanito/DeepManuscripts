package main.java;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.linalg.Vector;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONArray;

import scala.Tuple2;

import java.io.IOException;
import java.util.List;

public class CandidatePatchesMain {
	
	public static void main(String[] args) {
		String inputFile1, inputFile2, outputFile, dirRegex;
        SparkConf conf;
        JavaSparkContext sc;
        if(args[0].equals("--local")) {
            conf = new SparkConf().setAppName("DeepManuscript preprocessing").setMaster("local");
            sc = new JavaSparkContext(conf);
            inputFile1 = args[1];
            inputFile2 = args[2];
            dirRegex = args[3];
            outputFile = args[4];
        }else {
            conf = new SparkConf().setAppName("DeepManuscript preprocessing");
            sc = new JavaSparkContext(conf);
            inputFile1 = args[0];
            inputFile2 = args[1];
            dirRegex = args[2];
            outputFile = args[3];
        }

        JavaPairRDD<String, ImageData> dataImages = loadImg(sc, inputFile1, dirRegex);
        JavaPairRDD<String, String> dataJSON = loadJSON(sc, inputFile2, dirRegex);
        JavaPairRDD<String, Tuple2<ImageData,String>> data=  dataImages.join(dataJSON);
        JavaRDD<Tuple2<Vector,Vector>> patches = data.flatMap(new FlatMapFunction<Tuple2<String,Tuple2<ImageData,String>>, Tuple2<Vector,Vector>>(){
        	
			@Override
			public List<Tuple2<Vector, Vector>> call(
					Tuple2<String, Tuple2<ImageData, String>> arg0)
					throws Exception {
				List<Tuple2<Vector, Vector>> result = ExtractPatches.processLines(arg0._1(),arg0._2());
				return result;
			}
        });
        patches.saveAsTextFile(outputFile);
	}
	
	public static JavaPairRDD<String,ImageData> loadImg(JavaSparkContext sc, String inputFolder, String regex) {
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
                String filename = data._1().substring(data._1().length()-22,data._1().length() -4);
                //String basename = FilenameUtils.removeExtension(filename);
                return new Tuple2<String, ImageData>(filename, new ImageData(data._2().toArray()));
            }
        });
        return dataImages;
    }
	
	public static JavaPairRDD<String,String> loadJSON(JavaSparkContext sc, String inputFolder, String regex) {
        //Get a handle for every file in the directory
        JavaPairRDD<String,String> dataStream  = sc.wholeTextFiles(FilenameUtils.concat(inputFolder, regex))
                //Filter non-image file
                .filter(new Function<Tuple2<String, String>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<String, String> d) throws Exception {
                        String ext = FilenameUtils.getExtension(d._1());
                        return ext.equals("json") || ext.equals("JSON") ;
                    }
                });

        final String fInputFileAbs = findAbsolutePath(sc,inputFolder);
        //Remove the input path and extract ImageData from the PortableDataStream
        JavaPairRDD<String, String> dataJSON = dataStream.mapToPair(new PairFunction<Tuple2<String, String>, String, String>() {
            public Tuple2<String, String> call(Tuple2<String, String> data) {
                String filename = data._1().substring(data._1().length()-23,data._1().length()-5);
                //String basename = FilenameUtils.removeExtension(filename);
                return new Tuple2<String, String>(filename, data._2());
            }
        });
        return dataJSON;
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
