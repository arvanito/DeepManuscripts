package main.java;






import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;


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

	private AutoencoderConfig conf;
	
	public Autoencoder(AutoencoderConfig conf){
		this.conf = conf;
		
		this.rho = conf.getRho();
		this.lambda = conf.getLambda();
		this.beta = conf.getBeta();
		
		this.alpha = conf.getBeta();
		this.momentum = conf.getMomentum();
		this.initMomentum = conf.getMomentum();
		this.increaseMomentum = conf.getIncreaseMomentum();
		
		this.numEpochs = conf.getNumEpochs();
		this.numBatches = conf.getNumBatches();
		
		this.num_hidden = conf.getNum_hidden();
	}
	
	public Vector[] train(JavaRDD<Vector> data){

		sc = JavaSparkContext.fromSparkContext(data.context());
		num_input = data.take(1).iterator().next().size();
		
		
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

				AutoencoderGradient3 gradient = new AutoencoderGradient3(brParams,batches[j],conf);
				AutoencoderFctGrd    result = gradient.getGradient();

//				AutoencoderFct autoencoderFct = new AutoencoderFct(brParams, testData, conf);
//				double testError = autoencoderFct.computeTestError();

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

		return AutoencoderLinAlgebra.getFilters(params);
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

	

	

	

	

}
