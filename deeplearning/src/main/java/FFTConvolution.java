//package main.java;
//
//public class FFTConvolution {
//	/** TRUE uses internal naïve FFT impl, FALSE uses Fft.java:transform(). */ 
//	private static final boolean USE_INTERNAL = false;
//
//	/**
//	 * A naïve recursive implementation of the Cooley-Tukey FFT algorithm.
//	 * http://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
//	 * 
//	 * @param x The values to be FFT'd
//	 * @param N length of x
//	 * @param s kind of offset, should be 1 in the beginning
//	 * @param o offset, should be 0 in the beginning
//	 * @return A 2d double array with the resulting complex numbers
//	 */
//	private static double[][] fftCooleyTukeyRec(double[][] x, int N, int s, int o) {
//		double[][] res = new double[N][2];
//		if (N == 1) {
//			res[0] = x[o].clone();
//		} else {
//			double[][] a = fftCooleyTukeyRec(x, N/2, 2*s, o);
//			double[][] b = fftCooleyTukeyRec(x, N/2, 2*s, o+s);
//			for (int k = 0; k <= N/2 -1; ++k) {
//				double[] ta = a[k];
//				double[] tb = b[k];
//				double er = Math.cos(-2*Math.PI*k/N);
//				double ei = Math.sin(-2*Math.PI*k/N);
//				double r = er*tb[0] - ei*tb[1];
//				double i = er*tb[1] + ei*tb[0];
//				res[k][0] = ta[0] + r;
//				res[k][1] = ta[1] + i;
//				res[k+N/2][0] = ta[0] - r;
//				res[k+N/2][1] = ta[1] - i;
//			}
//		}
//		return res;
//	}
//	
//	/**
//	 * A proxy to call fftCooleyTukeyRec. Maybe extend to choose between different FFT-implementations?
//	 * For example http://www.nayuki.io/res/free-small-fft-in-multiple-languages/Fft.java
//	 * 
//	 * @param x An array of (real,img)-pairs to be transformed 
//	 * @return A transformed array of (real,img)-pairs
//	 */
//	private static double[][] fft1D(double [][] x) {
//		if(USE_INTERNAL) {
//			return fftCooleyTukeyRec(x, x.length, 1, 0);
//		} else {
//			Fft.transform(x);
//			return x;
//		}
//	}
//	
//	
//	/**
//	 * 1D (rows==1) or 2D FastFourierTransform
//	 * 
//	 * @param x real matrix to be FFT'd
//	 * @return complex matrix
//	 */
//	public static double[][][] fftReal1Dor2D(double[][] x) {
//		int rows = x.length;
//		int cols = x[0].length;
//
//		double[][][] y = new double[rows][][];
//		
//		for(int j = 0; j < rows; ++j) {
//			y[j] = fftReal1D(x[j]); 
//		}
//		
//		if(rows == 1) { /* 1D */
//			return y;
//		}
//		
//		double[][][] yt = transpose(y);
//		double[][][] z = new double[cols][][];
//		
//		for(int i = 0; i < cols; ++i) {
//			z[i] = fft1D(yt[i]); 
//		}
//
//		return transpose(z);
//	}
//	
//	/**
//	 * A proxy to call fft with real vector
//	 * 
//	 * @param x real vector to be FFT'd 
//	 * @return complex vector
//	 */
//	static double[][] fftReal1D(double[] x) {
//		return fft1D(realToComplex(x));
//	}
//	
//	
//	/**
//	 * Add zero imaginary part to a real vector
//	 * @param x real vector
//	 * @return complex vector
//	 */
//	private static double[][] realToComplex(double[] x) {
//		double[][] y = new double[x.length][2];
//		for(int i = 0; i < x.length; ++i) {
//			y[i][0] = x[i];
//		}
//		return y;
//	}
//	
//	/**
//	 * Drop imaginary part of a complex vector
//	 * @param x complex vector
//	 * @return real vector
//	 */
//	private static double[] complexToReal(double[][] x) {
//		double[] y = new double[x.length];
//		for(int i = 0; i < x.length; ++i) {
//			y[i] = x[i][0];
//		}
//		return y;
//	}
//	
//	/**
//	 * Drop imaginary part of a complex matrix
//	 * @param x complex matrix
//	 * @return real matrix
//	 */
//	private static double[][] complexToReal(double[][][] x) {
//		int rows = x.length;
//		int cols = x[0].length;
//		double[][] y = new double[rows][cols];
//		for(int j = 0; j < rows; ++j) {
//			for(int i = 0; i < cols; ++i) {
//				y[j][i] = x[j][i][0];
//			}
//		}
//		return y;
//	}
//	
//	/**
//	 * Swap real and imaginary parts of a vector
//	 * @param x imaginary vector
//	 * @return imaginary vector with real and imaginary parts swapped
//	 */
//	private static double[][] swappedRealImg(double[][] x) {
//		double[][] y = new double[x.length][2];
//		for(int i = 0; i < x.length; ++i) {
//			y[i][1] = x[i][0];
//			y[i][0] = x[i][1];
//		}
//		return y;
//	}
//	
//	/**
//	 * Inverse FastFourierTransform, using the fft()-function.
//	 * 
//	 * @param x complex array to be inverse-FFT'd
//	 * @return transformed complex array
//	 */
//	private static double[][] ifft(double [][] x) {
//		double[][] y = fft1D(swappedRealImg(x));
//		for (int i = 0; i < y.length; ++i) {
//			y[i][0] /= y.length;
//			y[i][1] /= y.length;
//		}
//		return swappedRealImg(y);
//	}
//	
//	/**
//	 * Inverse 1D (rows==1) or 2D FastFourierTransform, using the fft()-function.
//	 * 
//	 * @param x complex matrix to be inverse-FFT'd
//	 * @return transformed complex matrix
//	 */
//	private static double[][][] ifft(double[][][] x) {
//		int rows = x.length;
//		int cols = x[0].length;
//
//		double[][][] y = new double[rows][][];
//		
//		for(int j = 0; j < rows; ++j) {
//			y[j] = ifft(x[j]); 
//		}
//		
//		if(rows == 1) { /* 1D */
//			return y;
//		}
//		
//		double[][][] yt = transpose(y);
//		double[][][] z = new double[cols][][];
//		
//		for(int i = 0; i < cols; ++i) {
//			z[i] = ifft(yt[i]); 
//		}
//
//		return transpose(z);
//	}
//	
//	/**
//	 * Transposes the given 2D complex matrix
//	 * @param x 2D complex matrix
//	 * @return 2D complex matrix (transposed)
//	 */
//	private static double[][][] transpose(double[][][] x) {
//		int rows = x.length;
//		int cols = x[0].length;
//		double[][][] y = new double[cols][rows][2];
//		for(int j = 0; j < rows; ++j) { // looping over X's rows
//			for(int i = 0; i < cols; ++i) { // looping over X's cols
//				y[i][j][0] = x[j][i][0];
//				y[i][j][1] = x[j][i][1];
//			}
//		}
//		return y;
//	}
//	
//	/**
//	 * 1D complex dot-product
//	 * @param x 1D complex x
//	 * @param y 1D complex y of same size as x
//	 * @return 1D complex result of same size as x
//	 */
//	private static double[][] dotprod(double[][] x, double[][] y) {
//		if(x.length != y.length) throw new RuntimeException("Input vectors have different lengths");
//		double[][] z = new double[x.length][2];
//		for(int i = 0; i < x.length; ++i) {
//			z[i][0] = x[i][0]*y[i][0] - x[i][1]*y[i][1];
//			z[i][1] = x[i][0]*y[i][1] + x[i][1]*y[i][0];
//		}
//		return z;
//	}
//	
//	/**
//	 * 2D complex dot-product
//	 * @param x 2D complex x
//	 * @param y 2D complex y of same size as x
//	 * @return 2D complex result of same size as x
//	 */
//	private static double[][][] dotprod(double[][][] x, double[][][] y) {
//		if(x.length != y.length) throw new RuntimeException("Input matrices have different number of rows");
//		if(x[0].length != y[0].length) throw new RuntimeException("Input matrices have different number of columns");
//		
//		int rows = x.length;
//		int cols = x[0].length;
//		double[][][] z = new double[rows][cols][2];
//		for(int j = 0; j < rows; ++j) {
//			for(int i = 0; i < cols; ++i) {
//				z[j][i][0] = x[j][i][0]*y[j][i][0] - x[j][i][1]*y[j][i][1];
//				z[j][i][1] = x[j][i][0]*y[j][i][1] + x[j][i][1]*y[j][i][0];
//			}
//		}
//		return z;
//	}
//	
//	/**
//	 * Convolution in 1D with pre-calculated FFT's. The FFT's must be of the same length, and the pre-FFT inputs must be real.
//	 * 
//	 * @param fftx 
//	 * @param ffty
//	 * @return real vector of the convolution result
//	 */
//	static double[] convolveWithPreFFT(double[][] fftx, double ffty[][]) {
//		return complexToReal(ifft(dotprod(fftx, ffty)));
//	}
//	
//	/**
//	 * Convolution in 1-2D with pre-calculated FFT's. The FFT's must be of the same length, and the pre-FFT inputs must be real.
//	 * 
//	 * @param fftx 
//	 * @param ffty
//	 * @return real matrix of the convolution result
//	 */
//	static double[][] convolveWithPreFFT(double[][][] fftx, double ffty[][][]) {
//		return complexToReal(ifft(dotprod(fftx, ffty)));
//	}
//}
