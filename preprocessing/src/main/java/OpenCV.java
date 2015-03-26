package main.java;

import org.apache.spark.SparkFiles;
import org.opencv.core.Core;

/**
 * Helper class to load OpenCV library
 */
public class OpenCV {
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
}
