package main.java;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

public class ExtractPatches implements FlatMapFunction<Tuple2<Vector, Vector>	, Tuple2<Vector, Vector>> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -8714737488467653555L;
	
	private int[] vecSize;		// original size of the flat vector
	private int[] patchSize; 	// patch size in each direction 
	
	/**
	 * Constructor to initialize vector and patch size.
	 * 
	 * @param vecSize Original size of the flat vector
	 * @param patchSize Patch size to extract from the flat vector
	 */
	public ExtractPatches(int[] vecSize, int[] patchSize) {
		this.vecSize = Arrays.copyOf(vecSize, vecSize.length);
		this.patchSize = Arrays.copyOf(patchSize, patchSize.length);
	}
	
	
	@Override
	public List<Tuple2<Vector,Vector>> call(Tuple2<Vector, Vector> v) {

		// extract the meta-data for the vector
		Vector meta = v._1;
		
		// reshape the input vector that corresponds to the data
		Vector vd = v._2;
		DenseMatrix M = MatrixOps.reshapeVec2Mat((DenseVector) vd, vecSize);
		
		// allocate memory for the final output Matrix 
		int blockSizeTotal = patchSize[0] * patchSize[1];
		int[] sizeSmall = {vecSize[0]-patchSize[0]+1, vecSize[1]-patchSize[1]+1};
		int numPatches = sizeSmall[0] * sizeSmall[1];

		List<Tuple2<Vector, Vector>> patchList = new ArrayList<Tuple2<Vector, Vector>>(numPatches);
		
		// main loop for patch extraction
		// TODO use a step size for the patch extraction process, NOT all overlapping patches!!!
		int countDim = 0;
		for (int j = 0; j < sizeSmall[1]; j++) {
			for (int i = 0; i < sizeSmall[0]; i++) {	
				
				double[] out = new double[blockSizeTotal];
				
				// loop over the block
				for (int l = 0; l < patchSize[1]; l++) {
					for (int k = 0; k < patchSize[0]; k++) {
						out[countDim] = M.apply(i+k,j+l);;
						countDim++;
					}
				}
				countDim = 0;
				
				// add the current extracted patch to the list together with its meta-data
				// the meta-data gets replicated
				patchList.add(new Tuple2<Vector, Vector>(meta, Vectors.dense(out)));
			}
		}
		
		return patchList;
	}
	
}

