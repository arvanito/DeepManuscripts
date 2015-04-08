package main.java;

import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Static class for image processing related functions
 */
public class ImageFunctions {	
    /**
     * Extract patches from a given image. Size is defined by configuration file.
     * @param imgMat ImageData containing the image to be processed
     * @param nbPatches Number of patches to be extracted from the image
     * @return list of extracted patches represented in a vector form
     */
    static public List<Vector> extractPatches(Mat imgMat, int nbPatches) {
        List<Vector> results = new ArrayList<>();

        int numRows = imgMat.rows();
        int numCols = imgMat.cols();
        
        System.out.println("Number of rows: " + numRows + " and number of columns: " + numCols);
        
        //TODO get size from config file
        int patchRows = 32;
        int patchCols = 32;
        
        //extraction loop
        for(int n=0;n<nbPatches;n++) {
            double[] patchData = new double[patchRows*patchCols];
            byte[] lineData = new byte[numCols];
            //find random left top corner
            int iRow = (int)Math.round(Math.random()*(numRows-patchRows));
            int iCol = (int)Math.round(Math.random()*(numCols-patchCols));
            
            //extract from the corner line by line
            for(int i=0;i<patchRows;i++) {
                //extract line from OpenCV
                imgMat.get(iRow + i, iCol, lineData);
                
                //cast from byte to double
                //TODO TO BE IMPROVED
                for(int j=0;j<patchCols;j++)
                    patchData[i+patchRows*j]=(lineData[j]>=0? lineData[j] : 256+lineData[j]);
            }
            //
            results.add(new DenseVector(patchData));
        }
        return results;
    }
}
