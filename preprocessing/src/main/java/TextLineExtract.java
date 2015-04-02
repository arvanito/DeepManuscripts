package main.java;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.io.Serializable;
import java.util.ArrayList;


public class TextLineExtract {
	
	public ArrayList<ImageData> extractTextLine(ImageData originalImg, Double smooth, Integer s, Double sigma, Integer off)
	{
		//Decompress the image
		Mat img = originalImg.getImage();
		Mat img_edge = new Mat(); //edge map
		Mat medial_seams []; //First element: indices, second element: row coordinates of medial seams (L)
		Mat sep_seams; //Coordinates of separating seams
		ArrayList<ImageData> text_lines; //The final text line patches
			
		Imgproc.Canny(img, img_edge, 100, 200, 3,true);
		
		medial_seams = computeMedialSeams(img, img_edge, smooth, s, off);

		sep_seams = computeSeparatingSeams(img, medial_seams[1], sigma, off);

		text_lines = extractPatches(sep_seams, medial_seams[0], off);
		
		return text_lines;
	}
	
	public Mat [] computeMedialSeams(Mat img, Mat edgeMap, Double smoothing, Integer s, Integer off)
	{
		Size imgSize = edgeMap.size();
		Double imgHeight = imgSize.height;
		Double imgWidth = imgSize.width;
		Double sliceWidth = Math.floor(imgWidth/Double.valueOf(s));
		Mat imgBin = new Mat(imgHeight.intValue(), imgWidth.intValue(), CvType.CV_8U);
		//Compute horizontal projection profiles for all edge image slices
		int k = 1;
		for(int i = 1; i < s+1 ; i++)
		{
			
		}
		return null;
		
	}
	
	public Mat computeSeparatingSeams(Mat img, Mat L, Double sigma, Integer off)
	{
		return null;
		
	}
	
	public ArrayList<ImageData> extractPatches(Mat sep_seam_ind, Mat medial_seam_ind, Integer off)
	{
		return null;
		
	}
	public static void main(String[] args)
	{
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		Mat img = Highgui.imread("/home/isinsu/Desktop/DeepManuscripts/preprocessing/input/man1.jpg");
		Mat img_edge = new Mat(); //edge map
		System.out.println(Double.valueOf(3));
		Imgproc.Canny(img, img_edge, 100, 200, 3,true);
		Highgui.imwrite("/home/isinsu/Desktop/DeepManuscripts/preprocessing/output/edgeMap.jpg",img_edge);
	}

}
