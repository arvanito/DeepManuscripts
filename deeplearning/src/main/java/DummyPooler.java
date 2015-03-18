package main.java;

import org.apache.spark.mllib.linalg.Vector;

/**
 * A dummy Pooler - just passes the data through.
 * 
 * @author Arttu Voutilainen
 *
 */
public class DummyPooler implements Pooler {

	private static final long serialVersionUID = 1505267555549652215L;

	@Override
	public Vector call(Vector data) throws Exception {
		return data;
	}

}
