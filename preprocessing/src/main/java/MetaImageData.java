package main.java;

import java.io.Serializable;


/**
 * Representation of class containing both image data and metadata.
 * It has ImageData and Metadata objects that are public.
 * We can access following from ImageData and Metadata
 * ImageData - getStatus
 * Metadata - x_coordinate, y-coordinate, page_number, page_name, line_number
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
     * Simple Constructor from image in byte format 
     * @param data- byte information about the image
     */
    public MetaImageData(byte[] data) {
    	this.imgdata = new ImageData(data);
    	this.metadata = new Metadata();
    }
    
    /**
     * Constructor from byte image and metadata
     * @param data - Image in byte format
     * @param _metadata - metadata object
     */
    public MetaImageData(byte[] data, Metadata _metadata) {
        this.imgdata = new ImageData(data);
        this.metadata = (Metadata)(_metadata.clone());
    }

}
