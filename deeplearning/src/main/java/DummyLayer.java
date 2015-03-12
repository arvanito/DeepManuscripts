package main.java;

/**
 * 
 * A dummy implementation of a Layer. Just passes the data through, using Dummy(Learner,Extractor,Pooler).
 *
 * @author Arttu Voutilainen
 * 
 */
public class DummyLayer extends BaseLayer {

	public DummyLayer() {
		super(new DummyLearner(), new DummyExtractor(), new DummyPooler());
	}

}
