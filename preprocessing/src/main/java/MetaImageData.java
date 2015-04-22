package main.java;


import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;


/**
 * Representation of some image data.
 * Because this class will be used by Spark RDD, it needs to implement Serializable.
 * Two representations are kept, Compressed and Uncompressed. This should be transparent to external classes, but should allow
 * to drastically reduce drive and/or network bandwidth if data is moved.
 */
public class MetaImageData implements Serializable {
   /**
	 * 
	 */
	private static final long serialVersionUID = 2867139312558954772L;

    //Status of the representation
        
    public ImageData imgdata;
    public Metadata metadata;

    /**
     * Constructor Simple
     */
    public MetaImageData(byte[] data) {
    	this.imgdata = new ImageData(data);
    	this.metadata = new Metadata();
    }

    public MetaImageData(byte[] data, Metadata _metadata) {
        this.imgdata = new ImageData(data);
        this.metadata = (Metadata)(_metadata.clone());
    }

}
