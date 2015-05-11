package main.java;




import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.DenseMatrix;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;




import scala.Tuple2;

public class Autoencoder {
	
private JavaRDD<Vector> trainData;
private JavaRDD<Vector> testData;

private double rho = 0.05;
private double lambda = 0.001;
private double beta = 6;

private double alpha = 0.0005;
private double momentum = 0.5;
private double initMomentum = 0.5;
private double increaseMomentum = 0.9;

private int numEpochs = 2;
private int numBatches = 2;

private int num_input = 32*32;
private int num_hidden = 32*32/2;

private AutoencoderParams params;
private AutoencoderParams oldGrad;

private JavaSparkContext sc;

public void train(JavaRDD<Vector> data){
	
	sc = JavaSparkContext.fromSparkContext(data.context());
	//Split to train and test data
	split(data);
	
	params = initializeWeights();
	for (int i=0;i<numEpochs;i++){
		//suffle randomly data
		JavaRDD<Vector>[] batches = splitBatches(trainData);
		
		momentum = initMomentum;
		
		oldGrad = null;
		for (int j=0;j<numBatches;j++){
			if(j==20){
				momentum = increaseMomentum;
			}
			Broadcast<AutoencoderParams> brParams = sc.broadcast(params);
			
			AutoencoderGradient3 gradient = new AutoencoderGradient3(brParams,batches[j]);
			AutoencoderFctGrd    result = gradient.getGradient();
			
			//double testError = computeTestError(brParams);
			
			if (oldGrad == null){
				oldGrad = AutoencoderLinAlgebra.updateInitial(result,alpha);
			}else{
				oldGrad = AutoencoderLinAlgebra.update(oldGrad,result,alpha,momentum);
			}
			
			params = AutoencoderLinAlgebra.updateParams(params,oldGrad);
			
			System.out.println("Epoch "+i+", batch "+j+" train="+result.getValue());//+" test="+testError);
		}
		
		alpha = alpha / 2.0;
	}
	
}

private AutoencoderParams initializeWeights(){
	return AutoencoderLinAlgebra.initialize(num_input,num_hidden);
}

private void split(JavaRDD<Vector> data){
	JavaRDD<Vector>[] splits = data.randomSplit(new double[]{0.8,0.2}, System.currentTimeMillis());
	trainData = splits[0];
	testData  = splits[1];
}

private JavaRDD<Vector>[] splitBatches(JavaRDD<Vector> data){
	double[] arraySplit = new double[numBatches];
	double part = 1.0/numBatches;
	for(int i=0;i<numBatches;i++){
		if (i==(numBatches-1)){
			arraySplit[i] = 1 - part*(numBatches-1);
		}else{
			arraySplit[i] = part;
		}
	}
	return data.randomSplit(arraySplit,System.currentTimeMillis());
}

private double computeTestError(Broadcast<AutoencoderParams> params){
	
	FirstLayerActivation firstLayerActivation = new FirstLayerActivation(params);
	JavaRDD<Vector> a2 = testData.mapPartitions(firstLayerActivation); 	
	
	ComputeRho computeRho = new ComputeRho();
	Vector rho_h =  a2.reduce(computeRho);
					
	long numSamples = testData.count();
	
	double[] rhoH = rho_h.toArray();
	double[] sparsityArray = new double[rhoH.length];
	
	double KL = 0;
	double t1,t2,t3,t4;
	for (int i=0;i<rhoH.length;i++){
		
		//rhoH might be negative
		try{
			t1 = rho * Math.log(rho/rhoH[i]*numSamples);
		}
		catch (Exception e){
			t1 = 0;
		}
		
		//1 case would be rhoH = 1;
		try{
			t2 = (1-rho) * Math.log((1-rho)/(1-rhoH[i]/numSamples));
		}
		catch (Exception e){
			t2 = 0;
		}
		
		KL += t1+t2;
		
		try{
			t3 = -rho/rhoH[i]*numSamples; 
		}catch (Exception e){
			t3 = 0;
		}
		
		try{
			t4 = (1-rho)/(1-rhoH[i]/numSamples); 
		}catch (Exception e){
			t4 = 0;
		}
		sparsityArray[i] = beta*(t3+t4);
		
	}
	
	Broadcast<AutoencoderComputedParams> computedBroadcast = sc.broadcast(new AutoencoderComputedParams(numSamples, sparsityArray));

	double norm1 = Vectors.norm((Vector)new DenseVector(params.getValue().getW1().toArray()),2);
	double norm2 = Vectors.norm((Vector)new DenseVector(params.getValue().getW2().toArray()),2);

	Compute compute = new Compute(params, computedBroadcast);
	AutoencoderSum  	sum     = new AutoencoderSum();
	

	double fctValue = testData.mapPartitions(compute).reduce(sum);
	

	return fctValue+0.5*lambda*(norm1*norm1+norm2*norm2)+beta*KL;
	
}

private static class Compute implements FlatMapFunction<Iterator<Vector>,Double>{
	private Broadcast<AutoencoderParams> params;
	private Broadcast<AutoencoderComputedParams> comp;
	
	public Compute(Broadcast<AutoencoderParams> params,Broadcast<AutoencoderComputedParams> comp){
		this.params = params;
		this.comp = comp;
	}

	@Override
	public Iterable<Double> call( Iterator<Vector> arg0)
			throws Exception {
		
		double fctValue = 0;
		
		while(arg0.hasNext()){
		
		double[] x  = arg0.next().toArray();
		double[] a2A = AutoencoderLinAlgebra.MAM(params.getValue().getW1(), x);					
		for(int i=0;i<a2A.length;i++){
			a2A[i] = AutoencoderLinAlgebra.sigmoid(a2A[i]+params.getValue().getB1().apply(i));
			//Avoid complex computation and use a Singleton class to get precomputed values
			//AutoencoderSigmoid.getInstance().getValue(values[i]);
			
		}
		
							
		
		
		double[] delta3A = AutoencoderLinAlgebra.VAA(params.getValue().getB2(),
				AutoencoderLinAlgebra.MAM(params.getValue().getW2(),a2A)); 
		for(int i=0;i<delta3A.length;i++){
			double sig = AutoencoderLinAlgebra.sigmoid(delta3A[i]);
				//Avoid complex computation and use a Singleton class to get precomputed values
			//AutoencoderSigmoid.getInstance().getValue(values[i]);
			double val = (x[i]-sig);
			fctValue += val*val;			
		}

		
		};
		
		
		
		ArrayList<Double> result = new ArrayList<Double>();
		result.add(0.5*fctValue/comp.getValue().getNumSamples());
		return result;
	}
	
}

/*private static class AutoencoderFct implements Function<Vector,Double>{
	
	Broadcast<AutoencoderParams> params;
	private Broadcast<AutoencoderComputedParams> comp;
	
	public AutoencoderFct(Broadcast<AutoencoderParams> params, Broadcast<AutoencoderComputedParams> comp){
		this.params = params;
		this.comp = comp;
	}
	@Override
	public Double call(Vector arg0) throws Exception {
		
		int num_input = params.getValue().getW1().numCols();
		int num_hidden = params.getValue().getW1().numRows();
		long num_samples = comp.getValue().getNumSamples();
		
		double[] x  = arg0.toArray();
		double[] a2A = AutoencoderLinAlgebra.MVM(params.getValue().getW1(), arg0);					
		for(int i=0;i<a2A.length;i++){
			a2A[i] = AutoencoderLinAlgebra.sigmoid(a2A[i]+params.getValue().getB1().apply(i));
			//Avoid complex computation and use a Singleton class to get precomputed values
			//AutoencoderSigmoid.getInstance().getValue(values[i]);
			
		}
		
							
		double fctValue =0;
		
		double[] delta3A = AutoencoderLinAlgebra.VAA(params.getValue().getB2(),
				AutoencoderLinAlgebra.MAM(params.getValue().getW2(),a2A)); 
		double[] b3A = new double[delta3A.length];
		for(int i=0;i<delta3A.length;i++){
			double sig = AutoencoderLinAlgebra.sigmoid(delta3A[i]);
				//Avoid complex computation and use a Singleton class to get precomputed values
			//AutoencoderSigmoid.getInstance().getValue(values[i]);
			double val = (x[i]-sig);
			fctValue += val*val;	
			b3A[i] = delta3A[i]/num_samples;
		}
		
		return fctValue;
	}
	
}
*/

private static class AutoencoderSum implements Function2<Double,Double,Double>{

	@Override
	public Double call(Double arg0, Double arg1) throws Exception {
		// TODO Auto-generated method stub
		return arg0+arg1;
	}
	
}

private static class FirstLayerActivation implements FlatMapFunction<Iterator<Vector>,Vector>{
	
	
	private Broadcast<AutoencoderParams> params;
	
	public FirstLayerActivation(Broadcast<AutoencoderParams> params) {
		this.params = params;
	}

	
	public Iterable<Vector> call(Iterator<Vector> arg0) throws Exception {
//		double[] b1V = params.getValue().getB1().toArray();
//		DenseVector result = new DenseVector(params.getValue().getB1().toArray());
//		BLAS.gemv(1.0, params.getValue().getW1(), (DenseVector) arg0._1, 0.0, result);
		double[] values = new double[params.getValue().getW1().numRows()];
		while(arg0.hasNext()){
		double[] newValue = AutoencoderLinAlgebra.MVM(params.getValue().getW1(), arg0.next());
		for(int i=0;i<values.length;i++){
			values[i] += AutoencoderLinAlgebra.sigmoid(newValue[i]+params.getValue().getB1().apply(i));
			//Avoid complex computation and use a Singleton class to get precomputed values
			//AutoencoderSigmoid.getInstance().getValue(values[i]);
		}
		}
		ArrayList<Vector> result = new ArrayList<Vector>();
		result.add(new DenseVector(values));
		return result;
	}
	
}


private static class ComputeRho implements Function2<Vector, Vector, Vector>{

	@Override
	public Vector call(Vector arg0,  Vector arg1) throws Exception {
//		DenseVector result = new DenseVector(arg0._1.toArray());
//		BLAS.axpy(1.0, arg1._1, result);
		return new DenseVector(AutoencoderLinAlgebra.VVA(arg0,arg1)); 
	}
	
}

}
