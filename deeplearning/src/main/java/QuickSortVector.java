package main.java;

import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

public class QuickSortVector {
	
	 static int partition(Vector arr[], int left, int right)
	{
	      int i = left, j = right;
	      Vector tmp;
	      Vector pivot = arr[(left + right) / 2];
	     
	      while (i <= j) {
	            while (compareLeft(arr[i],pivot))
	                  i++;
	            while (compareRight(arr[j],pivot))
	                  j--;
	            if (i <= j) {
	                  tmp = new DenseVector( arr[i].toArray());
	                  arr[i] = new DenseVector(arr[j].toArray());
	                  arr[j] = new DenseVector(tmp.toArray());
	                  i++;
	                  j--;
	            }
	      };
	     
	      return i;
	}
	 
	public static void  quickSort(Vector arr[], int left, int right) {
	      int index = partition(arr, left, right);
	      if (left < index - 1)
	            quickSort(arr, left, index - 1);
	      if (index < right)
	            quickSort(arr, index, right);
	}
	
	static boolean compareRight(Vector i,Vector j){
		double[] iA = i.toArray();
		double[] jA = j.toArray();
		int idx=0;
		int len=i.size();
		boolean result = true;
		while(idx<len & result){
			if (iA[idx]<=jA[idx]){
				result = false;
			}
			idx++;
		}
		return result;
	}
	
	static boolean compareLeft(Vector i,Vector j){
		double[] iA = i.toArray();
		double[] jA = j.toArray();
		int idx=0;
		int len=i.size();
		boolean result = true;
		while(idx<len & result){
			if (iA[idx]>=jA[idx]){
				result = false;
			}
			idx++;
		}
		return result;
	}
}
