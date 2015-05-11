/**
 * 
 */
package main.java;

import java.util.Arrays;

import main.java.DeepModelSettings.ConfigBaseLayer;
import main.java.DeepModelSettings.ConfigFeatureExtractor;

import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

/**
 * @author blizzara
 *
 */
public class FFTConvolutionExtractor implements Extractor {

	private static final long serialVersionUID = -6605790709668749617L;

	private DenseMatrix zca;
	private DenseVector mean;
	
	double[][][][] featureFFTs = null; // 1. features 2. rows 3. cols 4. real/img
	double[] featureAdds = null;
	
	private int inputRows;
	private int inputCols;
	private int featureRows;
	private int featureCols;
	private int validRows;
	private int validCols;
	
	private ConfigFeatureExtractor.NonLinearity nonLinearity = null;  
	private double alpha; // non-linearity (threshold)
	
	public FFTConvolutionExtractor(ConfigBaseLayer configLayer) {
		setConfig(configLayer);
	}
	
	@Override
	public void setPreProcessZCA(DenseMatrix zca, DenseVector mean) {
		this.zca = zca;
		this.mean = mean;
	}
	
	/* (non-Javadoc)
	 * @see main.java.Extractor#setFeatures(org.apache.spark.mllib.linalg.Vector[])
	 */
	@Override
	public void setFeatures(Vector[] features) {
		/**
		 * Should pre-process features, flip 'em, pad them to match input data size and calculate their FFT's
		 */
		featureFFTs = new double[features.length][][][];
		
		if(zca != null && mean != null) {
			featureAdds = new double[features.length];
		}
		
		for(int i = 0; i < features.length; ++i) {
			DenseVector feature = (DenseVector)features[i];
			if(zca != null && mean != null) {
				DenseVector temp = new DenseVector(new double[features[0].size()]);
				//System.out.println("before: "+feature);
				BLAS.gemv(1.0, zca.transpose(), feature, 0.0, temp); // "de-whiten" features, instead of whitening data
				feature = temp;
				featureAdds[i] = -BLAS.dot(mean,feature); // the effect of the mean vector - to be added to each cell
				//System.out.println("after: "+feature + "\nmean: " + mean  + "\nadd: " + featureAdds[i]);
			}
			featureFFTs[i] = FFT(pad(inputRows, inputCols, flip(vectorToMatrix(featureRows, featureCols, feature.toArray()))));
		}
	}
	
	private void setConfig(ConfigBaseLayer configLayer) {
		ConfigFeatureExtractor conf = configLayer.getConfigFeatureExtractor(); 
		inputCols = conf.getInputDim1();
		inputRows = conf.getInputDim2();
		if(inputCols == 0 || (inputCols & (inputCols-1)) != 0) throw new RuntimeException("Configured input dimension 1 is 0 or not power-of-two");
		if(inputCols == 0 || (inputRows & (inputRows-1)) != 0) throw new RuntimeException("Configured input dimension 2 is 0 or not power-of-two");
		
		featureCols = conf.getFeatureDim1();
		featureRows = conf.getFeatureDim2();
		if(featureCols == 0 || featureCols > inputCols) throw new RuntimeException("Configured feature dimension 1 is 0 or > input dimension 1");
		if(featureRows == 0 || featureRows > inputRows) throw new RuntimeException("Configured feature dimension 2 is 0 or > input dimension 2");
		
		validRows = inputRows - featureRows + 1;
		validCols = inputCols - featureCols + 1;
		nonLinearity = conf.getNonLinearity();
		System.out.println(nonLinearity);
		if (conf.hasSoftThreshold()) {
			alpha = conf.getSoftThreshold();
		}
		
	}

	/* (non-Javadoc)
	 * @see main.java.Extractor#call(org.apache.spark.mllib.linalg.Vector)
	 */
	@Override
	public Vector call(Vector data) throws Exception {
		if(data.size() != inputRows*inputCols) throw new RuntimeException("Vector length does not match config");
		assert featureFFTs != null: "Features must be set before call()!";
		
		double[] result = new double[(validRows*validCols)*featureFFTs.length];
		int resultPos = 0;
		double[][][] dataFFT = FFT(vectorToMatrix(inputRows, inputCols, data.toArray()));
		for(int i = 0; i < featureFFTs.length; ++i) {
			double[][][] featureFFT = featureFFTs[i];
			double add = featureAdds != null ? featureAdds[i] : 0.0;
			double[] tmp = matrixToVector(ConvolveWithPreFFTandStrip(dataFFT, featureFFT), add);
			System.arraycopy(tmp, 0, result, resultPos, tmp.length);
			resultPos += tmp.length;
		}
		DenseVector res = new DenseVector(result);
		if(nonLinearity != null) res = MatrixOps.applyNonLinearityVec(res, nonLinearity, alpha);
		return res;
	}
	
	/**
	 * Does FFT for a real vector whose dimension is a power of 2.
	 * 
	 * @param in A Vector to be FFT'd
	 * @return A transformed array of (real,img)-pairs
	 */
	private double[][][] FFT(double[][] x) {
		return FFTConvolution.fftReal1Dor2D(x);
	}

	/**
	 * Does Convolution for two FFT arrays, handles 
	 * @param vx FFT'd x vector, of power-of-2 size
	 * @param vy FFT'd y vector, of length smaller than equal to vx
	 * @return A real array containing the result (non-zero-padded part of it)
	 */
	private double[][] ConvolveWithPreFFTandStrip(double[][][] fftx, double[][][] ffty) {
		assert fftx.length == ffty.length: "Convolve: row count not equal";
		assert fftx[0].length == ffty[0].length: "Convolve: col count not equal";
		//System.out.println(Arrays.deepToString(fftx));
		//System.out.println(Arrays.deepToString(ffty));
		
		double[][] convolved = FFTConvolution.convolveWithPreFFT(fftx,ffty);
		//System.out.println(Arrays.deepToString(convolved));
		int rows = convolved.length;
		int cols = convolved[0].length;
		double[][] stripped = new double[validRows][];
		
		for(int i = 0; i < validRows; ++i) {
			stripped[i] = Arrays.copyOfRange(convolved[rows-validRows+i], cols-validCols, cols);
		}
		//System.out.println(Arrays.deepToString(stripped));
		return stripped;
	}
	
	/**
	 * Converts from row-major vector to matrix (a11 a12 a13 a21 a22 a23..) => (a11 a12 a13; a21 a22 a23;...)
	 * @param rows number of rows in the matrix
	 * @param cols number of cols in the matrix
	 * @param x the vector containing the values
	 * @return a (rows,cols)-matrix containing the values
	 */
	private double[][] vectorToMatrix(int rows, int cols, double[] x) {
		if(rows*cols != x.length) throw new RuntimeException("The wanted matrix size differs from the vector length");
		
		double[][] y = new double[rows][cols];
		for(int j = 0; j < rows; ++j) {
			for(int i = 0; i < cols; ++i) {
				y[j][i] = x[i+j*cols];
			}
		}
		return y;
	}
	
	/**
	 * Converts from matrix to row-major vector  (a11 a12 a13; a21 a22 a23;...) => (a11 a12 a13 a21 a22 a23..)
	 * @param x the matrix containing the values
	 * @param add value to be added to each cell
	 * @return a row-major vector containing the values
	 */
	private double[] matrixToVector(double[][] x, double add) {
		int rows = x.length;
		int cols = x[0].length;
		double[] y = new double[rows*cols];
		for(int j = 0; j < rows; ++j) {
			for(int i = 0; i < cols; ++i) {
				y[i+j*cols] = x[j][i] + add;
			}
		}
		return y;
	}
	
	/**
	 * Pads a matrix with zeros to right and below to match given size
	 * 
	 * @param rows
	 * @param cols
	 * @param x
	 * @return
	 */
	private double[][] pad(int rows, int cols, double[][] x) {
		int xrows = x.length;
		
		double[][] y = new double[rows][];
		for(int j = 0; j < xrows; ++j) {
			y[j] = Arrays.copyOf(x[j], cols); // the rest is filled with zeros 
		}
		for(int j = xrows; j < rows; ++j) {
			y[j] = new double[cols]; // the rest is filled with zeros
		}
		return y;
	}
	
	/**
	 * Flips a matrix: (a11 a12; a21 a22) => (a22 a21; a12 a11)
	 * @param x matrix to flip
	 * @return flipped matrix
	 */
	private double[][] flip(double[][] x) {
		int rows = x.length;
		int cols = x[0].length;
		double[][] y = new double[rows][cols];
		for(int j = 0; j < rows; ++j) { // looping over X's rows
			for(int i = 0; i < cols; ++i) { // looping over X's cols
				y[j][i] = x[rows-j-1][cols-i-1];
			}
		}
		return y;
	}

}