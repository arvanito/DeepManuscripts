import java.io.Serializable;

import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.DenseVector;

/* Class that is used for gathering input arguments for map and reduce functions.

Available methods:
	- Config: constructor for assigning values to variables
	- setDims: sets the dimensions of the input
	- setFilters: set the learned filters
	- setZCA: set the ZCA Matrix
	- setMean: set the mean patch
	- setPoolSize: set the pool size
	- setRfSize: set the receptive field size
	- setK: set the number of filters
	- setEps1: set the regularizer for contrast normalization
	- setEps2: set the regularizer for ZCA whitening

	- getDims: get the dimensions of the input
	- getFilters: get the learned filters
	- getZCA: get the ZCA Matrix
	- getMean: get the mean patch
	- getPoolSize: get the pool size
	- getRfSize: get the receptive field size
	- getK: get the number of filters
	- getEps1: get the regularizer for contrast normalization 
	- getEps2: get the regularizer for ZCA whitening

*/

public class Config implements Serializable {

	// private variables that make up the input configuration
	private int[] dims;
	private DenseMatrix filters;
	private DenseMatrix zca;
	private DenseVector m;
	private int[] poolSize;
	private int[] rfSize;
	private int k;
	private Double eps1;
	private Double eps2;
	private int numGroups;
	private Integer[] groups;

	// default constructor
	public Config() {}

	// constructor that initializes the private variables
	public Config(int[] dimsIn, DenseMatrix filtersIn, DenseMatrix zcaIn, DenseVector mIn, int[] poolSizeIn, int[] rfSizeIn, int kIn, Double eps1In, Double eps2In, int numGroupsIn, Integer[] groupsIn) {
		dims = dimsIn;
		filters = filtersIn;
		zca = zcaIn;
		m = mIn;
		poolSize = poolSizeIn;
		rfSize = rfSizeIn;
		k = kIn;
		eps1 = eps1In;
		eps2 = eps2In;
		numGroups = numGroupsIn;
		groups = groupsIn;
	}


	/* set functions */

	// set input image dimensions
	public void setDims(int[] dimsIn) {
		dims = dimsIn;
	}

	// set learned filters
	public void setFilters(DenseMatrix filtersIn) {
		filters = filtersIn;
	}

	// set ZCA matrix
	public void setZCA(DenseMatrix zcaIn) {
		zca = zcaIn;
	}

	// set mean patch
	public void setMean(DenseVector mIn) {
		m = mIn;
	}

	// set size of the pooling block
	public void setPoolSize(int[] poolSizeIn) {
		poolSize = poolSizeIn;
	}

	// set size of the receptive field
	public void setRfSize(int[] rfSizeIn) {
		rfSize = rfSizeIn;
	}

	// set number of filters
	public void setK(int kIn) {
		k = kIn;
	}

	// set eps1 for contrast normalization
	public void setEps1(Double eps1In) {
		eps1 = eps1In;
	}

	// set eps2 for ZCA whitening
	public void setEps2(Double eps2In) {
		eps2 = eps2In;
	}

	// set number of groups
	public void setNumGroups(int numGroupsIn) {
		numGroups = numGroupsIn;
	}

	// set groups from K-means on learned filters
	public void setGroups(Integer[] groupsIn) {
		groups = groupsIn;
	}

	/* get functions */

	// get input image dimensions
	public int[] getDims() {
		return dims;
	}

	// get learned filters
	public DenseMatrix getFilters() {
		return filters;	
	}

	// get ZCA matrix
	public DenseMatrix getZCA() {
		return zca;
	}

	// get the mean of the training patches
	public DenseVector getMean() {
		return m;
	}

	// get size of pooling block
	public int[] getPoolSize() {
		return poolSize;
	}

	// get size of receptive field
	public int[] getRfSize() {
		return rfSize;
	}

	// get number of filters
	public int getK() {
		return k;
	}

	// get eps1 for contrast normalization
	public Double getEps1() {
		return eps1;
	}

	// get eps2 for ZCA whitening
	public Double getEps2() {
		return eps2;
	}

	// get number of groups
	public int getNumGroups() {
		return numGroups;
	}

	// get groups from K-means on learned filters
	public Integer[] getGroups() {
		return groups;
	}

}