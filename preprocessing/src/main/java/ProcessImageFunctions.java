package main.java;

import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by benoit on 25/03/2015.
 */
public class ProcessImageFunctions {
    /**
     * Extract patches from a given image. Size is defined by configuration file.
     * @param img ImageData containing the image to be processed
     * @param nbPatches Number of patches to be extracted from the image
     * @return list of extracted patches represented in a vector form
     */
    static public List<Vector> extractPatches(ImageData img, int nbPatches) {
        List<Vector> results = new ArrayList<Vector>();
        //TODO get size from config file
        Mat imgMat = img.getImage();
        int imgHorizontalSize = imgMat.rows();
        int imgVerticalSize = imgMat.cols();
        int patchHorizontalSize = 5;
        int patchVerticalSize = 5;
        double[] patchData = new double[patchHorizontalSize*patchVerticalSize];
        byte[] lineData = new byte[patchHorizontalSize];
        //extraction loop
        for(int n=0;n<nbPatches;n++) {
            //find random left top corner
            int iCol = (int)Math.round(Math.random()*(imgHorizontalSize-patchHorizontalSize));
            int iRow = (int)Math.round(Math.random()*(imgVerticalSize-patchVerticalSize));
            //extract from the corner line by line
            for(int i=0;i<patchVerticalSize;i++) {
                //extract line from OpenCV
                imgMat.get(iRow + i, iCol, lineData);
                //cast from byte to double
                for(int j=0;j<patchHorizontalSize;j++)
                    patchData[i*patchHorizontalSize+j]=(double)lineData[j];
            }
            //
            results.add(new DenseVector(patchData));
        }
        return results;
    }
}
