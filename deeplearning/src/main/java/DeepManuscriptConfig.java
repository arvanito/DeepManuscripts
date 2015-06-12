package main.java;

import java.util.List;

import main.java.DeepModelSettings.ConfigBaseLayer;

public class DeepManuscriptConfig {
	
	// type of arguments in the parameter list
	private String runningMode;
	private List<ConfigBaseLayer> protoBufConfig;
	private String inputDataset1;
	private String inputDataset2;
	private String queryDataset;
	private String testDataset;
	private String pathPrefixTrain;
	private String pathPrefixTest;
	
	
	/**
	 * Constructor that assigns all the necessary variables to the configuration.
	 * 
	 * @param runningMode Running mode: train, test, rank.
	 * @param protoBufConfig Protobuf configuration.
	 * @param inputDataset1 First input dataset for training.
	 * @param inputDataset2 Second input dataset for training.
	 * @param queryDataset Query dataset for ranking.
	 * @param testDataset Test dataset for ranking/testing.
	 * @param pathPrefixTrain Path prefix for saving the trained model.
	 * @param pathPrefixTest Path prefix for saving test output.
	 */
	public DeepManuscriptConfig(String runningMode, List<ConfigBaseLayer> protoBufConfig, String inputDataset1, String inputDataset2, 
								String queryDataset, String testDataset, String pathPrefixTrain, String pathPrefixTest) {
		this.setRunningMode(runningMode);
		this.setProtoBufConfig(protoBufConfig);
		this.setInputDataset1(inputDataset1);
		this.setInputDataset2(inputDataset2);
		this.setQueryDataset(queryDataset);
		this.setTestDataset(testDataset);
		this.setPathPrefixTrain(pathPrefixTrain);
		this.setPathPrefixTest(pathPrefixTest);
	}

	/**
	 * Get the running mode.
	 * 
	 * @return Running mode.
	 */
	public String getRunningMode() {
		return runningMode;
	}


	/**
	 * Set the running mode.
	 * 
	 * @param runningMode Input running mode.
	 */
	public void setRunningMode(String runningMode) {
		this.runningMode = runningMode;
	}


	/**
	 * Get the protobuf configuration.
	 * 
	 * @return Protobuf configuration.
	 */
	public List<ConfigBaseLayer> getProtoBufConfig() {
		return protoBufConfig;
	}


	/**
	 * Set the protobuf configuration.
	 * 
	 * @param protoBufConfig Input protobuf configuation.
	 */
	public void setProtoBufConfig(List<ConfigBaseLayer> protoBufConfig) {
		this.protoBufConfig = protoBufConfig;
	}


	/**
	 * Get the first dataset for training.
	 * 
	 * @return First dataset for training.
	 */
	public String getInputDataset1() {
		return inputDataset1;
	}


	/**
	 * Set the first dataset for training.
	 * 
	 * @param inputDataset1 First input dataset for training.
	 */
	public void setInputDataset1(String inputDataset1) {
		this.inputDataset1 = inputDataset1;
	}


	/**
	 * Get the second dataset for training.
	 * 
	 * @return Second input dataset for training.
	 */
	public String getInputDataset2() {
		return inputDataset2;
	}


	/**
	 * Set the second dataset for training.
	 * 
	 * @param inputDataset2 Second input dataset for training.
	 */
	public void setInputDataset2(String inputDataset2) {
		this.inputDataset2 = inputDataset2;
	}


	/**
	 * Get the query dataset for ranking.
	 * 
	 * @return Query dataset for ranking.
	 */
	public String getQueryDataset() {
		return queryDataset;
	}


	/**
	 * Set the query dataset for ranking.
	 * 
	 * @param queryDataset Input query dataset for ranking.
	 */
	public void setQueryDataset(String queryDataset) {
		this.queryDataset = queryDataset;
	}


	/**
	 * Get the test dataset for testing/ranking.
	 * 
	 * @return Test dataset for testing/ranking.
	 */
	public String getTestDataset() {
		return testDataset;
	}


	/**
	 * Set the test dataset for testing/ranking.
	 * 
	 * @param testDataset Test dataset for testing/ranking.
	 */
	public void setTestDataset(String testDataset) {
		this.testDataset = testDataset;
	}


	/**
	 * Get the path prefix for saving the trained model.
	 * 
	 * @return Training path prefix.
	 */
	public String getPathPrefixTrain() {
		return pathPrefixTrain;
	}


	/**
	 * Set the path prefix for saving the trained model.
	 * 
	 * @param pathPrefixTrain Input training path prefix.
	 */
	public void setPathPrefixTrain(String pathPrefixTrain) {
		this.pathPrefixTrain = pathPrefixTrain;
	}


	/**
	 * Get the path prefix for saving the testing output.
	 * 
	 * @return Testing path prefix.
	 */
	public String getPathPrefixTest() {
		return pathPrefixTest;
	}


	/**
	 * Set the path prefix for saving the testing output.
	 * 
	 * @param pathPrefixTest Input testing path prefix.
	 */
	public void setPathPrefixTest(String pathPrefixTest) {
		this.pathPrefixTest = pathPrefixTest;
	}
	
	
}
