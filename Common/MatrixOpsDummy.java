import java.util.*;

import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.SparseMatrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;


/* helper class for base Matrix operations

Available methods:
	- transpose: returns the transpose of a Matrix
	- reshapeVec2Mat: reshape a ector to a Matrix
	- reshapeMat2Mat: reshape a Matrix to another Matrix
	- convertVectors2Mat: convert array of Vectors to a Matrix
	- vec: vectorize column-major a Matrix
	- getRow: returns a specific row from a Matrix 
	- getCol: returns a specific column from a Matrix
	- toString: overrides the toString() method of the classes Vector and Matrix

	- meanColVec: get mean column Vector of a Matrix
	- meanRowVec: get mean row Vector of a Matrix

	- VecNormSq: squared norm of a Vector
	- MatNormSq: squared norm of a Matrix 
	- VecDistSq: squared euclidean distance between two Vectors
	- MatDistSq: squared euclidean distance between two matrices
	- ComputeDistancesSq: Squared pair-wise distances between rows of Matrices

	- DiagMatMatMult: Multiplication of a Matrix by a diagonal Matrix from the left
	- MatMatMult: Matrix-Matrix multiplication
	- MatVecMult: Matrix-Vector multiplication
	- VecVecMultIn: Inner product between two Vectors
	- VecVecMultOut: Outer Vector product

	- im2col: compute all overlapping 2-d patches of a Matrix
	- conv2: 2-d convolution
	- localVecContrastNormalization: compute contrast normalization on a local Vector
	- localMatContrastNormalization: compute contrast normalization on a local Matrix, column-by-column
	- localVecSubtractMean: subtract mean from a local Vector
	- localMatSubtractMean: subtract mean from a local Matrix, row-by-row
	- pool: max-pool of a Matrix
*/
public class MatrixOps {

	// method that returns the transpose of a Matrix
	// update this method using the new update method of the DenseMatrix class
	public DenseMatrix transpose(DenseMatrix M) {
		
		// Matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the transpose
		double[] Mt = new double[n*m];

		// perform the transposition
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				Mt[j+m*i] = M.apply(i, j);
			}
		}

		return new DenseMatrix(m, n, Mt);
	}

	// method that computes the transpose of a sparse matrix
	// make it more efficient and take into account the more 
	// general case of non-square
	public SparseMatrix transposeSparse(SparseMatrix M) {

		// size of the matrix
		int n = M.numRows();
		int m = M.numCols();

		// create a deep copy of the sparse matrix
		SparseMatrix Mt = M.copy();

		// fill-in the tranpose sparse matrix
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				Mt.toArray()[j+n*i] = M.apply(i, j);
				//System.out.println("Row " + i + ", Col " + j + ": " + Mt.apply(i, j) + " " + M.apply(i, j));
			}
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				System.out.println("Row " + i + ", Col " + j + ": " + Mt.apply(i, j) + " " + M.apply(i, j));	
			}
		}

		return Mt;
	}
	
	
	// method that reshapes the input Vector to a Matrix with specified dimensions
	public DenseMatrix reshapeVec2Mat(DenseVector v, int[] dims) {

		return new DenseMatrix(dims[0], dims[1], v.toArray()); 
	}


	// method that reshapes the input Matrix to a Matrix
	public DenseMatrix reshapeMat2Mat(DenseMatrix M, int[] dims) {

		return new DenseMatrix(dims[0], dims[1], M.toArray());
	}


	// method that converts an array of Vectors to a Matrix
	public DenseMatrix convertVectors2Mat(Vector[] V, int k) {

		// size of the Vectors inside the array
		int s = V[0].size();

		// allocate memory for output Matrix
		double[] out = new double[s*k];
		
		// assign Vectors to each row of the Matrix
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < s; j++) {
				out[i+k*j] = V[i].apply(j);
			}
		}

		return new DenseMatrix(k, s, out);
	}


	// method that vectorizes column-major the input Matrix
	public DenseVector vec(DenseMatrix M) {

		// size of the Matrix
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the Vector
		double[] out = new double[m*n];

		// global counter of the Vector
		int c = 0;

		// main assignment loop 
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				out[c] = M.apply(i, j);
				c++;	
			}
		}

		return new DenseVector(out);
	}


	// get a specific row from a Matrix, index starts from zero
	public DenseVector getRow(DenseMatrix M, int r) throws IndexOutOfBoundsException {
		
		// Matrix size
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

	// get a specific row from a Matrix, index starts from zero
	public Vector getRowV(DenseMatrix M, int r) throws IndexOutOfBoundsException {
		
		// Matrix size
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

		return Vectors.dense(row);
	}

	// get several rows from a Matrix

	// get a specific column from a Matrix
	public DenseVector getCol(DenseMatrix M, int c) throws IndexOutOfBoundsException {
		
		// Matrix size
		int n = M.numRows();
		int m = M.numCols();

		// check for the index bounds
		if (c >= m) {
			throw new IndexOutOfBoundsException("Column index argument is out of bounds!");
		}

		// return the specified column
		double[] col = new double[n];
		for (int i = 0; i < n; i++) {
			col[i] = M.apply(i, c);
		}
		
		return new DenseVector(col);
	}

	// get several columns from a Matrix

	
	// get mean column Vector of a Matrix
	public DenseVector meanColVec(DenseMatrix M) {

		// Matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the mean Vector and temporary sum Vector
		double[] out = new double[n];

		// compute mean Vector
		double sum = 0.0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				sum += M.apply(i ,j);
			}
			out[i] = sum / m;
			sum = 0.0;
		}

		return new DenseVector(out);
	}


	// get mean row Vector of a Matrix
	public DenseVector meanRowVec(DenseMatrix M) {

		// Matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the mean Vector and temporary sum Vector
		double[] out = new double[m];

		// compute mean Vector
		double sum = 0.0;
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				sum += M.apply(i ,j);
			}
			out[j] = sum / n;
			sum = 0.0;
		}

		return new DenseVector(out);
	}


	// override toString() function for printing an Object of the DenseVector class
	public String toString(DenseVector v) {
		
		// Vector length
		int s = v.size();

		// StringBuilder allocates memory from before, better performance than 
		// appending the string every time
		StringBuilder out = new StringBuilder(s*32);
		for (int i = 0; i < s; i++) {
			out.append(v.apply(i));
			out.append("\t");	// tab between any element in the row
		}

		return out.toString();
	}


	// override toString() function for printing an Object of the DenseMatrix class
	public String toString(DenseMatrix M) {
		
		// Matrix size
		int n = M.numRows();
		int m = M.numCols();

		// StringBuilder allocates memory from before, better performance than 
		// appending the string every time
		StringBuilder out = new StringBuilder(n*m*32);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				out.append(M.apply(i, j));
				out.append("\t");	// tab between any element in the row
			}
			out.append("\n");	// newline for the next row
		}

		return out.toString();
	}


	// squared l2 norm of a Vector
	public double vecNormSq(DenseVector v) {
	
		// Vector length
		int s = v.size();

		// sum of squares
		double normSq = 0.0;
		for (int i = 0; i < s; i++) {
			normSq += v.apply(i) * v.apply(i);
		}

		return normSq;
	}


	// squared l2 norm of a Matrix (sum of squared values) 
	public double matNormSq(DenseMatrix M) {
	
		// size of the Matrix
		int n = M.numRows();
		int m = M.numCols();
		
		// sum of squares
		double normSq = 0.0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				normSq += M.apply(i, j) * M.apply(i, j);
			}
		}

		return normSq;
	}


	// compute squared l2 distance between two Vectors
	public double vecDistSq(DenseVector v1, DenseVector v2) throws IllegalArgumentException {
	
		// length of the first Vector
		int s1 = v1.size();

		// length of the second Vector, it should have the same size!
		int s2 = v2.size();

		if (s1 != s2) {
			throw new IllegalArgumentException("The two Vectors do not have the same length!");
		}

		// sum of element-wise squared differences
		double distSq = 0.0;
		for (int i = 0; i < s1; i++) {
			distSq += (v1.apply(i) - v2.apply(i)) * (v1.apply(i) - v2.apply(i));
		}

		return distSq;
	}


	// compute squared l2 distance between two matrices
	public double matDistSq(DenseMatrix A, DenseMatrix B) throws IllegalArgumentException {
	
		// dimensions of the first Matrix
		int n1 = A.numRows();
		int m1 = A.numCols();

		// dimension of the second Matrix, should be the same as the first!
		int n2 = B.numRows();
		int m2 = B.numCols();

		if ((n1 != n2) || (m1 != m2)) {
			throw new IllegalArgumentException("The two matrices do not have the same length!");
		}

		// sum of element-wise squared differences
		double distSq = 0.0;
		for (int i = 0; i < n1; i++) {
			for (int j = 0; j < m1; j++) {
				distSq += (A.apply(i, j) - B.apply(i, j)) * (A.apply(i, j) - B.apply(i, j));
			}
		}

		return distSq;
	}

	// compute sqaured distances between a Vector and a Matrix
	// the return argument will be a Vector with pair-wise distances
	public DenseVector computeDistancesVecMatSq(DenseVector v, DenseMatrix M) throws IllegalArgumentException {
		
		// size of the vector
		int s = v.size();
		
		// dimensions of the Matrix, the number of columns should be the same
		int n = M.numRows();
		int m = M.numCols();

		if (s != m) {
			throw new IllegalArgumentException("The vector and the matrix should have the same number of columns!");
		}

		// compute pair-wise distances
		double[] distSq = new double[n];
		for (int i = 0; i < n; i++) {
			distSq[i] = vecDistSq(v, getRow(M, i));
		}
		
		return new DenseVector(distSq);
	}


	// compute sqaured distances between row Vectors in two different matrices
	// the return argument will be a Matrix with pair-wise distances 
	public DenseMatrix computeDistancesMatMatSq(DenseMatrix A, DenseMatrix B) throws IllegalArgumentException {
		
		// dimensions of the first Matrix
		int n1 = A.numRows();
		int m1 = A.numCols();

		// dimensions of the second Matrix, the number of columns should be the same
		int n2 = B.numRows();
		int m2 = B.numCols();

		if (m1 != m2) {
			throw new IllegalArgumentException("The two matrices should have the same number of columns!");
		}

		// compute pair-wise distances
		double[] distSq = new double[n1*n2];
		for (int i = 0; i < n1; i++) {
			for (int j = 0; j < n2; j++) {
				distSq[i+n1*j] = vecDistSq(getRow(A, i), getRow(B, j));
			}
		}
		
		return new DenseMatrix(n1, n2, distSq);
	}


	// compute a sparse k-nn weight matrix from a matrix of data points
	public SparseMatrix computeWeightMatrix(DenseMatrix M, int k, double sigma) {

		// dimensions of the matrix
	 	int n = M.numRows();
	 	int m = M.numCols();

	 	// allocate memory for array of values and indices 
	 	int[] colPtr = new int[n+1];
	 	int[] rowIdx = new int[n*k];
	 	double[] values = new double[n*k];

	 	final Integer[] idx = new Integer[n];
	 	for (int i = 0; i < n; i++) {

	 		// get current row of the matrix
	 		DenseVector c = getRow(M, i);

	 		// compute distances of all points from this point
	 		final DenseVector dist = computeDistancesVecMatSq(c, M);

	 		// compute indices of the sorted distances
	 		for (int j = 0; j < n; j++) {
	 			idx[j] = j;
	 		}
	 		Arrays.sort(idx, new Comparator<Integer>() {
    			@Override public int compare(final Integer o1, final Integer o2) {
        			return Double.compare(dist.toArray()[o1], dist.toArray()[o2]);
    			}
			});

	 		// find only the first k indices and sort them again, so that 
	 		// the assigned values correspond to them
	 		int[] sortIdx = new int[k];
	 		for (int j = 0; j < k; j++) {
	 			sortIdx[j] = idx[j];
	 		}
	 		Arrays.sort(sortIdx);

	 		// assign values and indices
	 		int kk = 0;
	 		for (int j = i*k; j < (i+1)*k; j++) {
	 			rowIdx[j] = sortIdx[kk];
	 			values[j] = dist.toArray()[sortIdx[kk]];
	 			kk++;	
	 		}
	 		colPtr[i] = i * k;
	 	}
	 	colPtr[n] = n * k;

	 	// create sparse matrix
	 	SparseMatrix W = new SparseMatrix(n, n, colPtr, rowIdx, values);
	 	
	 	// make it symmetric

	 	return W;

 		// HERE: Construct either mutual or normal graph

	 	// Unweighted graph?

	 	// create a gaussian function from the euclidean distances
	}


	// spectral clustering using the graph Laplacian


	// multiply a diagonal Matrix (Vector in reality) with a Matrix
	public DenseMatrix diagMatMatMult(DenseVector v, DenseMatrix M) throws IllegalArgumentException {
	
		// size of the Matrix, Vector's size is the same as the Matrix rows
		int n = M.numRows();
		int m = M.numCols();

		// Vector length
		int s = v.size();
		
		// throw an exception if sizes are not compatible
		if (n != s) {
			throw new IllegalArgumentException("Incompatible Vector and Matrix sizes!");
		}

		// allocate memory for the output Matrix
		double[] out = new double[n*m];
	
		// perform the multiplication
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < n; i++) {
				out[i+n*j] = v.apply(i) * M.apply(i, j);
			}
		}

		return new DenseMatrix(n, m, out);
	}


	// multiply a Matrix with a Matrix from the left
	public DenseMatrix matMatMult(DenseMatrix A, DenseMatrix B) throws IllegalArgumentException  {
		
		// size of the first Matrix
		int n = A.numRows();
		int m = A.numCols();

		// columns of the second Matrix
		int p = B.numRows();
		int r = B.numCols();

		// throw an exception if Matrix sizes are not compatible
		if (m != p) {
			throw new IllegalArgumentException("Matrix sizes are incompatible!"); 
		}
	
		// allocate memory for the output Matrix
		double[] out = new double[n*r];
		
		// perform the multiplication
		double s = 0.0;
		for (int i = 0; i < n; i++ ) {
			for (int j = 0; j < r; j++) {
				for (int k = 0; k < m; k++) {
					s += A.apply(i, k) * B.apply(k, j);
				}
			
				// the final inner product is the resulting (i,j) entry
				out[i+n*j] = s;
				s = 0.0;
			}
		}
		
		return new DenseMatrix(n, r, out);
	}

	// Matrix Vector multiplication
	//public Matrix MatVecMult(Matrix A, DenseVector x) throws IllegalStateException {}
	
	// inner product between two Vectors
	//public DenseVector VecVecMultIn(DenseVector v1, DenseVector v2) throws IllegalStateException {}

	// outer product between two Vectors
	//public Matrix VecVecMultOut(DenseVector v1, DenseVector v2) throws IllegalStateException {}

	
	// compute all overlapping patches of a 2-D Matrix
	// each patch is a column of the resulting Matrix
	public DenseMatrix im2col(DenseMatrix M, int[] blockSize) {

		// size of the Matrix 
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the final output Matrix 
		// which contains in each column an overlapping patch
		int blockSizeTotal = blockSize[0] * blockSize[1];
		int[] sizeSmall = {n-blockSize[0]+1,  m-blockSize[1]+1};
		int numPatches = sizeSmall[0] * sizeSmall[1];
		
		double[] out = new double[blockSizeTotal*numPatches];

		// main loop for patch extraction
		int countPatch = 0;
		int countDim = 0;
		for (int j = 0; j < sizeSmall[1]; j++) {
			for (int i = 0; i < sizeSmall[0]; i++) {	

				// loop over the block
				for (int l = 0; l < blockSize[1]; l++) {
					for (int k = 0; k < blockSize[0]; k++) {
						out[countDim+blockSizeTotal*countPatch] = M.apply(i+k,j+l);
						countDim++;
					}
				}
				countPatch += 1;
				countDim = 0; 
			}
		}

		return new DenseMatrix(blockSizeTotal, numPatches, out);
	}


	// compute all overlapping patches of a 2-D Matrix
	// each patch is a row of the resulting matrix
	public DenseMatrix im2colT(DenseMatrix M, int[] blockSize) {

		// size of the Matrix 
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for the final output Matrix 
		// which contains in each column an overlapping patch
		int blockSizeTotal = blockSize[0] * blockSize[1];
		int[] sizeSmall = {n-blockSize[0]+1,  m-blockSize[1]+1};
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


	// 2-D convolution, without any zero-padding, resulting image of smaller size than original
	public DenseMatrix conv2(DenseMatrix M, DenseMatrix F) {

		// size of the Matrix to be convolved
		int n = M.numRows();
		int m = M.numCols();

		// size of kernel
		int r = F.numRows();
		int c = F.numCols();

		// allocate memory for the convolution result
		int oR = n - r + 1;
		int oC = m - c + 1;
		double[] out = new double[oR*oC];	

		// 2-D convolution
		double s = 0.0;
		for (int i = 0; i < oR; i++) {
			for (int j = 0; j < oC; j++) {

				// loop over the filter, multiply and sum up
				for (int k = 0; k < r; k++) {
					for (int l = 0; l < c; l++) {
						s += F.apply(k,l) * M.apply(i+k,j+l);
					}
				}
				out[i+oR*j] = s;
				s = 0.0;	
			}
		}

		return new DenseMatrix(oR, oC, out);
	}


	// compute contrast normalization on a local Vector
	public DenseVector localVecContrastNormalization(DenseVector v, double e) {	

		// Vector size
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
		//double e = 10;	// HERE change this!!!!
		for (int i = 0; i < s; i++) {
			v.toArray()[i] = v.apply(i) - m;
			v.toArray()[i] = v.apply(i) / Math.sqrt((stdev + e));
		}

		return v;
	}


	// compute contrast normalization on a local Matrix, column by column
	public DenseMatrix localMatContrastNormalization(DenseMatrix M, double e) {

		// Matrix size
		int n = M.numRows();
		int m = M.numCols();

		// allocate memory for temporary Vector and final Matrix
		DenseVector cur = new DenseVector(new double[m]);		
		DenseMatrix C = new DenseMatrix(n, m, new double[n*m]);

		// main loop for contrast normalization
		for (int i = 0; i < n; i++) {
			cur = getRow(M, i);
			cur = localVecContrastNormalization(cur, e);

			// copy the normalized row back to the result
			for (int j = 0; j < m; j++) {
				C.toArray()[i+n*j] = cur.toArray()[j];
			}
		}

		return C;
	}


	// subtract mean from a local Vector
	public DenseVector localVecSubtractMean(DenseVector v, DenseVector m) {

		// maybe here check the size of the Vector and throw an exception!!
		
		// Vector size
		int s = v.size();

		// loop over elements to subtract the two Vectors
		double[] sub = new double[s];
		for (int i = 0; i < s; i++) {
			sub[i] = v.apply(i) - m.apply(i);
		}
		
		return new DenseVector(sub);

	}


	// subtract mean from a local Matrix row by row
	public DenseMatrix localMatSubtractMean(DenseMatrix M, DenseVector v) {

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


	// max-pool an image
	public DenseMatrix pool(DenseMatrix M, int[] poolSize) {

		// Matrix size
		int n = M.numRows();
		int m = M.numCols();

		// pooled Matrix size
		int[] poolDim = new int[2];
		poolDim[0] = (int) Math.floor(n/poolSize[0]);
		poolDim[1] = (int) Math.floor(m/poolSize[1]);
		double[] out = new double[poolDim[0]*poolDim[1]];	
		
		// upper left patch indices
		int k = 0;
		int l = 0;
		
		// lower right patch indices		
		int kk = 0;
		int ll = 0;

		// main loop for 2-D max pooling
		for (int i = 0; i < poolDim[0]; i++) {
			for (int j = 0; j < poolDim[1]; j++) {

				// extract the current patch lower right indices
				kk = Math.min(k+poolSize[0],n);
				ll = Math.min(l+poolSize[1],m);

				// compute max in the current patch
				double maxPatch = -Double.MAX_VALUE;
				for (int p = k; p < kk; p++) {
					for (int q = l; q < ll; q++) {
						if (M.apply(p, q) > maxPatch) {
							maxPatch = M.apply(p, q);	
						}
					}
				}

				// assign the max value in the resulting Matrix
				out[i+poolDim[0]*j] = maxPatch;
				
				// go one step to the right
				l += poolSize[1];
			}
			
			// go one step down
			k += poolSize[0];
			l = 0;
		}

		return new DenseMatrix(poolDim[0], poolDim[1], out);
	}

	// group pooling over learned filters
	public DenseVector groupPool(DenseMatrix feats, int[] pooledDims, int k, Integer[] groups, int numGroups) {

		// dimensions of feature Matrix
		int n = feats.numRows();
		int m = feats.numCols();

		// allocate memory for the pooled features 
		double[] pooledFeats = new double[n*numGroups];
		for (int p = 0; p < n*numGroups; p++) {
			pooledFeats[p] = -Double.MAX_VALUE;
		}

		// max-pool the responses over the learned filters
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {

				// find current group of filters
				int g = groups[j];

				// update the pooled features
				double featsTemp = feats.apply(i, j); 
				if (featsTemp > pooledFeats[i+n*g]) {
					pooledFeats[i+n*g] = featsTemp;	
				}
			}
		}
		DenseVector pooledFeatures = new DenseVector(pooledFeats);

		return pooledFeatures;
	}

}

