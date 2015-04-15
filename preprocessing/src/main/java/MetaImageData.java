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
        
    public ImageData Imgdata;
    private Metadata metadata;

    //Probably other image metadata needed (compression type, compression level, precision of data, number of channels)
    
    /**
     * Constructor Simple
     */
    public MetaImageData(byte[] data,int x, int y, int w, int h) {
    	this.Imgdata = new ImageData(data);
    	this.metadata = new Metadata(x,y,w,h);
    }
    
    
    public void setX(int x) {
        this.metadata.x_coord = x;
    }
    
    public void setY(int x) {
        this.metadata.y_coord = x;
    }
    
    public void setWidth(int x) {
        this.metadata.width = x;
    }
    
    public void setHeight(int x) {
        this.metadata.height = x;
    }
    
    public int getX() {
    	if(this.metadata.x_coord !=(Integer) null)
    		return this.metadata.x_coord;
    	else {
    		System.out.println("X coordinate is NULL");
    		return (Integer) null;
    	}
    }
    
    public int getY() {
    	if(this.metadata.y_coord !=(Integer) null)
    		return this.metadata.y_coord;
    	else {
    		System.out.println("Y coordinate is NULL");
    		return (Integer) null;
    	}
    }
    
    public int getWidth() {
    	if(this.metadata.width !=(Integer) null)
    		return this.metadata.width;
    	else {
    		System.out.println("Width of the image is NULL");
    		return (Integer) null;
    	}
    }
    
    public int getHeight() {
    	if(this.metadata.height !=(Integer) null)
    		return this.metadata.height;
    	else {
    		System.out.println("Height of the image is NULL");
    		return (Integer) null;
    	}
    }

    /**
     * Serialization of Object
     * @param out
     * @throws IOException
     */
    public void writeObject(ObjectOutputStream out) throws IOException {
        //write normal fields
        out.defaultWriteObject();
        
    }

    /**
     * Serialization of Object
     * @param in
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public void readMetaObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        //read normal fields
        in.defaultReadObject();
        
    }

}
