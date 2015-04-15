package main.java;

import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Helper class for basic matrix manipulations.
 * 
 * Available methods:
 * 	- transpose:
 * 	- convertVectors2Mat:
 * 
 * @author Nikolaos Arvanitopoulos
 *
 */
public class MatrixOps {
	
	/** 
	 * Method that converts an array of Vectors to a DenseMatrix.
	 * 
	 * @param V Array of type Vector that represents the learned features
	 * @return A matrix of type DenseMatrix that contains in each row one learned feature
	 */
	public static DenseMatrix convertVectors2Mat(Vector[] V) {

		int k = V.length;
		
		int s = V[0].size();
		double[] out = new double[s*k];
		
		// assign Vectors to each row of the Matrix
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < s; j++) {
				out[i+k*j] = V[i].apply(j);
			}
		}

		//return Matrices.dense(k, s, out);
		return new DenseMatrix(k, s, out);
	}
	
	
	/**
	 * Method that extracts a row from a matrix.
	 * 
	 * @param M Input DenseMatrix
	 * @param r Row index
	 * @return Row of type DenseVector
	 * @throws IndexOutOfBoundsException
	 */
	public static DenseVector getRow(DenseMatrix M, int r) throws IndexOutOfBoundsException {
		
		int n = M.numRows();
		int m = M.numCols();

		// check for the index bounds
		if (r >= n) {
			throw new IndexOutOfBoundsException("Row index argument is out of bounds!");
		}

		// return the specified row
		double[] row = new double[m];
		for (int j = 0; j < m; j++) {
			row[j] = M.apply(r, j);
		}

		return new DenseVector(row);
	}
	
	
	/**
	 * Method that computes all overlapping patches of a matrix. 
	 * 
	 * @param M DenseMatrix to extract patches from
	 * @param blockSize Size of the extracted patches
	 * @return Matrix where each row corresponds to one extracted patch
	 */
	public static DenseMatrix im2colT(DenseMatrix M, int[] blockSize) {

		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the final output Matrix 
		int blockSizeTotal = blockSize[0] * blockSize[1];
		int[] sizeSmall = {n-blockSize[0]+1, m-blockSize[1]+1};
		int numPatches = sizeSmall[0] * sizeSmall[1];

		double[] out = new double[numPatches*blockSizeTotal];

		// main loop for patch extraction
		int countPatch = 0;
		int countDim = 0;
		for (int j = 0; j < sizeSmall[1]; j++) {
			for (int i = 0; i < sizeSmall[0]; i++) {	

				// loop over the block
				for (int l = 0; l < blockSize[1]; l++) {
					for (int k = 0; k < blockSize[0]; k++) {
						out[countPatch+numPatches*countDim] = M.apply(i+k,j+l);;
						countDim++;
					}
				}
				countPatch += 1;
				countDim = 0; 
			}
		}

		return new DenseMatrix(numPatches, blockSizeTotal, out);
	}
	
	
	/**
	 * Method that performs local matrix contrast normalization
	 * 
	 * @param M Input DenseMatrix
	 * @param e Parameter for contrast normalization
	 * @return Contrast normalized result
	 */
	public static DenseMatrix localMatContrastNorm(DenseMatrix M, double e) {

		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for temporary Vector and final Matrix
		DenseVector cur = new DenseVector(new double[m]);		
		DenseMatrix C = new DenseMatrix(n, m, new double[n*m]);

		// main loop for contrast normalization
		for (int i = 0; i < n; i++) {
			cur = getRow(M, i);
			cur = localVecContrastNorm(cur, e);

			// copy the normalized row back to the result
			for (int j = 0; j < m; j++) {
				C.toArray()[i+n*j] = cur.toArray()[j];
			}
		}

		return C;
	}
	
	
	/**
	 * Method that performs local matrix mean subtraction, row by row
	 * 
	 * @param M Input DenseMatrix for mean subtraction
	 * @param v Mean DenseVector
	 * @return Mean subtracted result
	 */
	public static DenseMatrix localMatSubtractMean(DenseMatrix M, DenseVector v) {

		// maybe here check the size of the Vector and throw an exception!!
		
		// Matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for temporary Vector and final Matrix
		DenseVector cur = new DenseVector(new double[m]);	
		DenseMatrix C = new DenseMatrix(n, m, new double[n*m]);

		// loop over elements to subtract the mean row by row
		for (int i = 0; i < n; i++) {
			cur = getRow(M, i);
			cur = localVecSubtractMean(cur, v);

			// copy the subtracted row back to the result
			for (int j = 0; j < m; j++) {
				C.toArray()[i+n*j] = cur.toArray()[j];
			}
		}

		return C;
	}
	
	
	/** 
	 * Method that performs contrast normalization on a local DenseVector.
	 * 
	 * @param v Input DenseVector to be processed
	 * @param e Parameter epsilon for the contrast normalization
	 * @return Contrast normalized Vector
	 */
	public static DenseVector localVecContrastNorm(DenseVector v, double e) {	

		int s = v.size();

		// compute mean value of the Vector
		double m = 0;
		for (int i = 0; i < s; i++) {
			m += v.apply(i);
		}
		m /= s;

		// compute standard deviation of the Vector
		double stdev = 0;
		for (int i = 0; i < s; i++) {
			stdev += (v.apply(i) - m) * (v.apply(i) - m);
		}
		stdev = stdev / (s - 1);

		// subtract mean and divide by the standard deviation
		for (int i = 0; i < s; i++) {
			v.toArray()[i] = v.apply(i) - m;
			v.toArray()[i] = v.apply(i) / Math.sqrt((stdev + e));
		}

		return v;
	}

	
	/** 
	 * Method that subtracts the mean from a local DenseVector.
	 * 
	 * @param v Input DenseVector to be processed
	 * @param m Mean DenseVector for subtraction
	 * @return Mean subtracted DenseVector
	 */
	public static DenseVector localVecSubtractMean(DenseVector v, DenseVector m) throws IllegalArgumentException {

		if (v.size() != m.size()) {
			throw new IllegalArgumentException("Vector sizes are incompatible!");
		}
		
		int s = v.size();

		// loop over elements to subtract the two Vectors
		double[] sub = new double[s];
		for (int i = 0; i < s; i++) {
			sub[i] = v.apply(i) - m.apply(i);
		}
		
		//return Vectors.dense(sub);
		return new DenseVector(sub);
	}
	
	
	/**
	 * Method that reshapes a matrix to a vector.
	 * 
	 * @param v Matrix of type DenseMatrix to be reshaped
	 * @return Output vector
	 */
	public static DenseVector reshapeMat2Vec(DenseMatrix M) {
		return new DenseVector(M.toArray()); 
	}
	
	/**
	 * Method that reshapes a vector to a matrix.
	 * 
	 * @param v Vector of type DenseVector to be reshaped
	 * @param dims Dimensions of the final matrix
	 * @return Reshaped matrix
	 */
	public static DenseMatrix reshapeVec2Mat(DenseVector v, int[] dims) {
		
		if (dims[0]*dims[1] != v.size()) {
			throw new IllegalArgumentException("Vector size and matrix dimensions are not compatible!");
		}
		
		return new DenseMatrix(dims[0], dims[1], v.toArray()); 
	}
	
	
	/**
	 * Method that transposes a DenseMatrix.
	 * 
	 * @param M Input DenseMatrix
	 * @return Transposed DenseMatrix
	 */
	public static DenseMatrix transpose(DenseMatrix M) {
		
		int n = M.numRows();
		int m = M.numCols();

		double[] Mt = new double[n*m];

		// perform the transposition
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				Mt[j+m*i] = M.apply(i, j);
			}
		}

		return new DenseMatrix(m, n, Mt);
	}
	
}
