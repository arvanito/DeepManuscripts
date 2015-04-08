package main.java;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

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
    private static final long serialVersionUID = -151442211116649858L;
    //Status of the representation
    public enum ImageDataState {UNCOMPRESSED, COMPRESSED, ERROR}
    private ImageDataState state;
    private String compression_type;

    //Probably other image metadata needed (compression type, compression level, precision of data, number of channels)
    

    /**
     * Constructor Simple
     */
    public MetaImageData() {
//        this.image = image; //Could be image.clone() for safety.
        state = ImageDataState.UNCOMPRESSED;
        compression_type = "XYZ";
    }
    /**
     * Constructor from ImageDataState
     */
    public MetaImageData(ImageDataState st) {
        //this.compressed_img = new MatOfByte(data);
        state = st;
        compression_type = "XYZ";
    }

    /**
     * Get the current compression status of the class.
     * @return current status.
     */
    public ImageDataState getStatus() {
        return state;
    }
    public String getCompressionType() {
    	return compression_type;
    }

    /**
     * Serialization of Object
     * @param out
     * @throws IOException
     */
    public void writeMetaObject(ObjectOutputStream out) throws IOException {
        //write normal fields
        //out.defaultWriteObject();
        out.writeObject(compression_type);
        out.writeObject(state);
        
    }

    /**
     * Serialization of Object
     * @param in
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public void readMetaObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        //read normal fields
        //in.defaultReadObject();
        compression_type = (String) in.readObject();
        state = (ImageDataState) in.readObject();
        
    }

}
