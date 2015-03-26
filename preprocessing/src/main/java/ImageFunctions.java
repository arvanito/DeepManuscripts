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
        int imgHorizontalSize = imgMat.cols();
        int imgVerticalSize = imgMat.rows();
        //TODO get size from config file
        int patchHorizontalSize = 9;
        int patchVerticalSize = 9;
        //extraction loop
        for(int n=0;n<nbPatches;n++) {
            double[] patchData = new double[patchHorizontalSize*patchVerticalSize];
            byte[] lineData = new byte[patchHorizontalSize];
            //find random left top corner
            int iCol = (int)Math.round(Math.random()*(imgHorizontalSize-patchHorizontalSize));
            int iRow = (int)Math.round(Math.random()*(imgVerticalSize-patchVerticalSize));
            //extract from the corner line by line
            for(int i=0;i<patchVerticalSize;i++) {
                //extract line from OpenCV
                imgMat.get(iRow + i, iCol, lineData);
                //cast from byte to double
                //TODO TO BE IMPROVED
                for(int j=0;j<patchHorizontalSize;j++)
                    patchData[i*patchHorizontalSize+j]=(lineData[j]>=0? lineData[j] : 256+lineData[j]);
            }
            //
            results.add(new DenseVector(patchData));
        }
        return results;
    }
}
