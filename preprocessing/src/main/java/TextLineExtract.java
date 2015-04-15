package main.java;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;


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
	
	public static Mat [] computeMedialSeams(Mat img, Mat edgeMap, Double smoothing, Integer s, Integer off)
	{
		Size imgSize = edgeMap.size();
		Double imgHeight = imgSize.height;
		Double imgWidth = imgSize.width;
		Integer sliceWidth = (int) Math.floor(imgWidth/Double.valueOf(s));
		Mat imgBin = new Mat(imgHeight.intValue()*imgWidth.intValue(), 1, CvType.CV_8UC1);
		ArrayList<ArrayList<Integer>> localMax = new ArrayList<ArrayList<Integer>>(s);
		
		//Compute horizontal projection profiles for all edge image slices
		Mat horizProj = new Mat(imgHeight.intValue(), 1, CvType.CV_32F);
		Mat horizProjSmooth = new Mat(1, imgHeight.intValue(), CvType.CV_32F);
		int k = 0;
		for(int i = 0; i < s ; i++)
		{
			//Sum
			
			for(int row = 0; row < imgHeight.intValue(); row ++)
			{
				Mat subRow = edgeMap.submat(row, row+1, k, k+sliceWidth-1);
				double [] rowSum  = new double[1];
				rowSum[0]= Core.sumElems(subRow).val[0];
				horizProj.put(row,0,rowSum[0]/255.0);
			}
			
			for(int fg = 0; fg < horizProj.size().height ; fg++)
			{
				double [] a = horizProj.get(fg, 0);
				//System.out.println(fg+1 + " "  +a[0]);
			}
			//Cubic Smoothing Spline
			for(int row = 0; row < imgHeight.intValue(); row ++)
			{
				//double rowSmoothSum = 0;
				//horizProjSmooth.put(1,row,rowSmoothSum);
				double [] pxl = horizProj.get(row,0);
				horizProjSmooth.put(0,row, pxl[0]);
			}
			//Find Peaks
			localMax.add(i,findPeaks(horizProjSmooth));
			
			k += sliceWidth;
		}
		
		k = (int) Math.floor(sliceWidth/2.0);
		ArrayList<Integer> medial_seams_ind = new ArrayList<Integer>();
		
		for(int i = 0; i < s ; i++)
		{
			//If there is no local max, continue to next pair
			if(localMax.get(i) == null || localMax.get(i).size() == 0)
				continue;
			
			//Matches from left to right and right to left
			ArrayList<Integer> matchesLeftToRight = new ArrayList<Integer>();
			ArrayList<Integer> matchesRightToLeft = new ArrayList<Integer>();
			
			//Find matches from left slice to right slice
			for(int j=0; j< localMax.get(i).size(); j++)
			{
				TreeMap<Double, Integer> dists = new TreeMap<Double, Integer>();
				for(int h = 0 ; h < localMax.get(i+1).size() ; h++)
				{
					int tempVal = Math.abs(localMax.get(i).get(j) - localMax.get(i+1).get(h));
					dists.put((double)tempVal, h);
				}
				matchesLeftToRight.add(dists.get(dists.firstKey()));
			}
			
			//Find matches from right slice to left slice
			for(int j=0; j< localMax.get(i+1).size(); j++)
			{
				TreeMap<Double, Integer> dists = new TreeMap<Double, Integer>();
				for(int h = 0 ; h < localMax.get(i+1).size() ; h++)
				{
					int tempVal = Math.abs(localMax.get(i+1).get(j) - localMax.get(i).get(h));
					dists.put((double)tempVal, h);
				}
				matchesRightToLeft.add(dists.get(dists.firstKey()));
			}
			
			//Match profile maxima that agree from both sides
			//(left to right and right to left)
			
			for(int j=0; j< localMax.get(i).size(); j++)
			{
				for(int o=0; o< localMax.get(i+1).size(); o++)
				{
					if(matchesLeftToRight.get(j) == o && matchesRightToLeft.get(o) == j)
					{
						ArrayList<Integer> inds = new ArrayList<Integer>();
						if(i == 1)
						{
							ArrayList<Integer> points = linspaceRound(localMax.get(i).get(j), localMax.get(i+1).get(o), k);
							
							for(int u = 0, g = 0 ; u < points.size() && g < k; u++, g++)
							{
								inds.add(sub2ind(imgHeight.intValue(), imgWidth.intValue(), u, g));
							
							}
							
						}
						else if(i == s-1)
						{
							int length = imgWidth.intValue() - k + 1;
							ArrayList<Integer> points = linspaceRound(localMax.get(i).get(j), localMax.get(i+1).get(o), length);
							for(int u = 0, g = k ; u < points.size() && g < imgWidth.intValue(); u++, g++)
							{
								inds.add(sub2ind(imgHeight.intValue(), imgWidth.intValue(), u, g));
								
							}
						
						}
						else
						{
							ArrayList<Integer> points = linspaceRound(localMax.get(i).get(j), localMax.get(i+1).get(o), sliceWidth);
							for(int u = 0, g = k; u < points.size() && g < k+sliceWidth-1; u++, g++)
							{
								inds.add(sub2ind(imgHeight.intValue(), imgWidth.intValue(), u, g));
								
							}
						}
						
						
						medial_seams_ind.addAll(inds);
						
						//update image bin
						for(int r = 0; r < inds.size() ; r++)
						{
							imgBin.put(inds.get(r), 1, 1);
						}
					}
				}
			}
			
			if(i > 1 && i < s-1)
			{
				k += sliceWidth;
			}
			
		}
		imgBin.reshape(0,imgHeight.intValue());
		Set<Integer> medial_seams_unique = new HashSet<Integer>();
		medial_seams_unique.addAll(medial_seams_ind);
		medial_seams_ind.clear();
		medial_seams_ind.addAll(medial_seams_unique);
		
		//post-processing step to remove lines from some intermediate column of the image
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(imgBin, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
		imgBin.reshape(0, imgHeight.intValue()*imgWidth.intValue());
		Integer numComp = contours.size();
		
		//remove medial seams that do not start from the beginning of the image
		for(int i = 0 ; i < numComp ; i++)
		{
			MatOfPoint coord = contours.get(i);
			List<Point> points = coord.toList();
			if((int)points.get(0).y != 1)
			{
				for(int j = 0; j < points.size() ; j++)
				{
					Integer linindx = sub2ind(imgHeight.intValue(), imgWidth.intValue(), (int)points.get(j).x, (int)points.get(j).y);
					int removeIndex = medial_seams_ind.indexOf(linindx);
					if(removeIndex > -1)
					{
						medial_seams_ind.remove(removeIndex);
					}
					imgBin.put(linindx, 1, 0);
				}
			}
		}
		imgBin.reshape(0, imgHeight.intValue());
		
		//post-processing step to extend the small lines towards the end column of the image
		contours.clear();
		Imgproc.findContours(imgBin, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
		numComp = contours.size();
		
		//extend medial seams to the end of the image, if possible
		for(int i = 0 ; i < numComp ; i++)
		{
			MatOfPoint coord = contours.get(i);
			List<Point> points = coord.toList();
			if((int)points.get(points.size()-1).y != imgWidth.intValue())
			{
				int endCol = (int)points.get(points.size()-1).y;
				ArrayList<Integer> indsExt = new ArrayList<Integer>();
				if(i == 0)
				{
					indsExt = linspaceFloor((int)points.get(points.size()-1).x, 1, imgWidth.intValue() - endCol + 1);
					for(int t = 0, u = endCol; t < indsExt.size() ; t++, u++)
					{
						Integer linIndsExt = sub2ind(imgHeight.intValue(), imgWidth.intValue(), indsExt.get(t), u);
						medial_seams_ind.add(linIndsExt);
					}
				}
				else if(i == numComp-1)
				{
					indsExt = linspaceFloor((int)points.get(points.size()-1).x, imgHeight.intValue() , imgWidth.intValue() - endCol + 1);
					for(int t = 0, u = endCol; t < indsExt.size() ; t++, u++)
					{
						Integer linIndsExt = sub2ind(imgHeight.intValue(), imgWidth.intValue(), indsExt.get(t), u);
						medial_seams_ind.add(linIndsExt);
					}
				}
				else
				{
					MatOfPoint coordPrev = contours.get(i-1);
					List<Point> pointsPrev = coordPrev.toList();
					
					MatOfPoint coordNext = contours.get(i+1);
					List<Point> pointsNext = coordNext.toList();
					
					int middle = (int)Math.floor((pointsPrev.get(pointsPrev.size()-1).x + pointsNext.get(pointsNext.size()-1).x) / 2);
					indsExt = linspaceFloor((int)points.get(points.size()-1).x, middle , imgWidth.intValue() - endCol + 1);
					
					ArrayList<Integer> inter_p = new ArrayList<Integer>();
					ArrayList<Integer> ip1 = new ArrayList<Integer>();
					ArrayList<Integer> ip2 = new ArrayList<Integer>();
					
					for(int index=0; index<pointsPrev.size(); index++)
					{
						if(indsExt.contains(pointsPrev.get(index).x))
						{
							inter_p.add((int)pointsPrev.get(index).x);
							ip1.add(indsExt.indexOf(pointsPrev.get(index).x));
							ip2.add(index);
						}
					}
					
					ArrayList<Integer> inter_n = new ArrayList<Integer>();
					ArrayList<Integer> in1 = new ArrayList<Integer>();
					ArrayList<Integer> in2 = new ArrayList<Integer>();
					
					for(int index=0; index<pointsNext.size(); index++)
					{
						if(indsExt.contains(pointsNext.get(index).x))
						{
							inter_n.add((int)pointsNext.get(index).x);
							in1.add(indsExt.indexOf(pointsNext.get(index).x));
							in2.add(index);
						}
					}
					
					if(!inter_p.isEmpty() && !inter_n.isEmpty())
					{
						if(indsExt.get(ip1.get(0)) < indsExt.get(in1.get(0)))
						{
							int tCopy = ip2.get(0) - endCol;
							while(true)
							{
								if(indsExt.size() > tCopy - 10)
									indsExt.remove(tCopy - 10);
								else
									break;
							}
							for(int t = 0, u = endCol; t < indsExt.size() ; t++, u++)
							{
								Integer linIndsExt = sub2ind(imgHeight.intValue(), imgWidth.intValue(), indsExt.get(t), u);
								medial_seams_ind.add(linIndsExt);
							}
						}
						else
						{
							int tCopy = in2.get(0) - endCol;
							while(true)
							{
								if(indsExt.size() > tCopy - 10)
									indsExt.remove(tCopy - 10);
								else
									break;
							}
							for(int t = 0, u = endCol; t < indsExt.size() ; t++, u++)
							{
								Integer linIndsExt = sub2ind(imgHeight.intValue(), imgWidth.intValue(), indsExt.get(t), u);
								medial_seams_ind.add(linIndsExt);
							}
						}
					}
					else if(!inter_p.isEmpty())
					{
						int tCopy = ip2.get(0) - endCol;
						while(true)
						{
							if(indsExt.size() > tCopy - 10)
								indsExt.remove(tCopy - 10);
							else
								break;
						}
						for(int t = 0, u = endCol; t < indsExt.size() ; t++, u++)
						{
							Integer linIndsExt = sub2ind(imgHeight.intValue(), imgWidth.intValue(), indsExt.get(t), u);
							medial_seams_ind.add(linIndsExt);
						}
					}
					else if(!inter_n.isEmpty())
					{
						int tCopy = in2.get(0) - endCol;
						while(true)
						{
							if(indsExt.size() > tCopy - 10)
								indsExt.remove(tCopy - 10);
							else
								break;
						}
						for(int t = 0, u = endCol; t < indsExt.size() ; t++, u++)
						{
							Integer linIndsExt = sub2ind(imgHeight.intValue(), imgWidth.intValue(), indsExt.get(t), u);
							medial_seams_ind.add(linIndsExt);
						}
					}
					else
					{
						for(int t = 0, u = endCol; t < indsExt.size() ; t++, u++)
						{
							Integer linIndsExt = sub2ind(imgHeight.intValue(), imgWidth.intValue(), indsExt.get(t), u);
							medial_seams_ind.add(linIndsExt);
						}
					}
				}
			}
		}
		
		imgBin = new Mat(imgHeight.intValue()*imgWidth.intValue(), 1, CvType.CV_8UC1);
		for(int l = 0; l < medial_seams_ind.size(); l++)
		{
			imgBin.put(medial_seams_ind.get(l), 1, 1);
		}
		
		imgBin.reshape(0, imgHeight.intValue());
		
		contours.clear();
		Imgproc.findContours(imgBin, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
		numComp = contours.size();
		
		ArrayList<ArrayList<Integer>> L = new ArrayList<ArrayList<Integer>>(numComp);
		
		for(int i = 0 ; i < numComp ; i++)
		{
			MatOfPoint coord = contours.get(i);
			List<Point> points = coord.toList();
			if(points.size() < 10)
			{
				continue;
			}
			
			/*MatOfPoint2f smoothedCurve = new MatOfPoint2f();
			MatOfPoint2f inpPoints = new MatOfPoint2f(coord.toArray());
			Imgproc.approxPolyDP(inpPoints, smoothedCurve, 0.1, true);*/
			ArrayList<Integer> seamRow = new ArrayList<Integer>();
 			for(int y = 0; y < points.size(); y++)
			{
				seamRow.add((int)points.get(y).x);
				System.out.println(seamRow.size());
			}
 			L.add(seamRow);
		}
		
		return null;
		
	}
	
	public static ArrayList<Integer> findPeaks(Mat inputArray){
		ArrayList<Integer> retArray = new ArrayList<Integer>();
		int index = 2;
		while(index < (int)inputArray.size().height-1)
		{
			double prevValue = inputArray.get(1, index-1)[0];
			double currentValue = inputArray.get(1,index)[0];
			double nextValue = inputArray.get(1,index+1)[0];
			if(currentValue > nextValue  && currentValue > prevValue)
			{
				retArray.add(index);
				index++;
			}
			index++;
		}
		return retArray;
	}
	
	public static ArrayList<Integer> linspaceRound(Integer elem1, Integer elem2, Integer range)
	{
		ArrayList<Integer> retArray = new ArrayList<Integer>();
		retArray.add(elem1);

		Double temp = (double) elem1;
		for(int i = 1 ; i < range + 1 ; i++)
		{
			if(elem1 > elem2)
			{
				temp -= Math.round(1.0/(double)range);
			}
			else
			{
				temp += Math.round(1.0/(double)range);
			}
			retArray.add(temp.intValue());
		}
		return retArray;
	}
	
	public static ArrayList<Integer> linspaceFloor(Integer elem1, Integer elem2, Integer range)
	{
		ArrayList<Integer> retArray = new ArrayList<Integer>();
		retArray.add(elem1);

		Double temp = (double) elem1;
		for(int i = 1 ; i < range + 1 ; i++)
		{
			if(elem1 > elem2)
			{
				temp -= Math.floor(1.0/(double)range);
			}
			else
			{
				temp += Math.floor(1.0/(double)range);
			}
			retArray.add(temp.intValue());
		}
		return retArray;
	}
	
	public static Integer sub2ind(Integer rowCount, Integer columnCount, Integer startIndex, Integer endIndex)
	{
		return rowCount*endIndex + startIndex;
	}
	
	public static Integer[] ind2sub(Integer rowCount, Integer columnCount, Integer ind)
	{
		Integer arr[] = new Integer[2];
		arr[1] = (int)Math.floor(ind/rowCount);
		arr[0] = ind%rowCount;
		return arr;
	}
	
	public static Mat computeSeparatingSeams(Mat img, Mat L, Double sigma, Integer off)
	{
			
		Mat img_blur = null;
		Double ksize = 50.0;
		Imgproc.GaussianBlur(img, img_blur, new Size(ksize, ksize), sigma);

		Mat FXR = null;
		Mat FYR = null;

		Mat kernelx = new Mat(1,3, CvType.CV_32F){
        		 {
        		   put(1,0,1);
        		 }
        	 };
         	Mat kernely = new Mat(3,1, CvType.CV_32F){
        	         {
       			   put(1,0,1);
        	         }
	         };
		Imgproc.filter2D(img_blur, FXR, -1, kernelx);
		Imgproc.filter2D(img_blur, FYR, -1, kernely);		
		
		Mat absFXR = null;
		Core.absdiff(FXR, Scalar.all(0), absFXR);
		Mat absFYR = null;
		Core.absdiff(FYR, Scalar.all(0), absFYR);
		Mat energy_map = null;
		Core.add(absFXR, absFYR, energy_map);
		Mat energy_map_t = energy_map.t();
				
		
		Size s = energy_map_t.size();
		Double n = s.height;
		Integer l = L.size();
		Mat sep_seams = Mat.zeros(new Size(n,l-1), CvType.CV_8U);
		Mat new_energy = energy_map_t;
				
		Integer i,row,col;
		for (i = 1; i < l; i++)
		{
			ArrayList<Integer> L_a = L.get(i);
			ArrayList<Integer> L_b = L.get(i+1);
		    
		    Integer l_a =  L_a.size();
		    Integer l_b =  L_b.size();
		    
		    Integer min_l = Math.min(l_a,l_b);
		    
		    for (row = 2; row <= min_l; row++)
		    {	    	
		    	for (col = L_a.get(row); col <=  L_b.get(row); col++)
		    	{
		            
		            Integer left = Math.max(col-1, L_a.get(row-1));
		            Integer right = Math.min(col+ 1, L_b.get(row+1));
		            
		            Mat temprow = new_energy.row(row-1).colRange(left, right);
		            Double minpath = Core.minMaxLoc(temprow).minVal;

		            byte tempbuff[] = new byte[(int) (new_energy.total() * new_energy.channels())];
		            if (minpath == null)
		            {
		            	if (col>left)
		            	{		            		
		            		new_energy.put(row, col, new_energy.get(row-1, right, tempbuff));
		            	}
		            	else if (col<right) 
		            	{
		            		new_energy.put(row, col, new_energy.get(row-1, left, tempbuff));							
				}
		            }
		            else
		            {
		            	new_energy.put(row, col, (new_energy.get(row-1, left, tempbuff)+minpath));
		            }		            	
		            
		    	}
		    }

		   Integer min_index = (int) Core.minMaxLoc(new_energy.row(min_l).colRange(L_a.get(min_l), L_b.get(min_l))).minLoc.y;
		   
		   min_index = min_index + L_a.get(min_l) - 1;

		   for (row = min_l-1; row>=1; row--)
		   {
			  Integer j = min_index;
			  
			  sep_seams.put(row+1, i, j);
			  
			  Integer left = Math.max(j-1, L_a.get(row));
			  Integer right = Math.min(j+1, L_b.get(row));

			  min_index = (int) Core.minMaxLoc(new_energy.row(row).colRange(left, right)).minLoc.y;
			  
			  if (min_index == null)
			  {
				  if (j>left)
				  {
					  min_index = right;
				  }
				  else if (j < right)
				  {
					  min_index = left;
				  }
			  }
			  else
			  {
				  min_index = min_index + left - 1;
			  }
		   }

		   sep_seams.put(l, i, min_index);
		   
		   new_energy = energy_map_t;
		   
		}
		
		return null;
		
	}
	
	public static ArrayList<ImageData> extractPatches(Mat sep_seam_ind, Mat medial_seam_ind, Integer off)
	{
		return null;
		
	}
	
	public static void main(String[] args)
	{
		System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
		Mat img = Highgui.imread("/home/isinsu/Desktop/DeepManuscripts/preprocessing/input/man1.jpg");
		
		Imgproc.cvtColor(img,img, Imgproc.COLOR_BGR2GRAY);
//		for(int i = 0; i < 800; i++)
//		{
//			double [] pxl = img.get(i,0);
//			System.out.println(pxl[0]);
//		}
		Size imgSize = img.size();
		Double imgHeight = imgSize.height;
		Double imgWidth = imgSize.width;
		Mat img_edge = new Mat(imgHeight.intValue(),imgWidth.intValue(),CvType.CV_32F); //edge map
		//System.out.println(Double.valueOf(3));
		Imgproc.Canny(img, img_edge, 100, 200, 3,true);
//		for(int i = 0; i < 800; i++)
//		{
//			for(int j = 0; j < 1855  ; j++)
//			{
//				double [] pxl2 = img_edge.get(453,j);
//				System.out.println(pxl2[0]);
//			}
//		}
		computeMedialSeams(img, img_edge, 0.0003, 8, 5);
		//Highgui.imwrite("/home/isinsu/Desktop/DeepManuscripts/preprocessing/output/edgeMap.jpg",img_edge);
	}

}
