package main.java;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;


/**
 * Representation of some image data parameters.
 * Two representations are kept, Compressed and Uncompressed. This should be transparent to external classes, but should allow
 * to drastically reduce drive and/or network bandwidth if data is moved.
 */
public class MetaImageData implements Serializable {
    //private static final long serialVersionUID = -151442211116649858L;
    private static final long serialVersionUID = 2867139312558954772L;
    //Status of the representation
    public enum ImageDataState {UNCOMPRESSED, COMPRESSED, ERROR}
    private ImageDataState state;
    private String compression_type;

    //Probably more image metadata can be added: PageNo, LineNo, X_coord, Y_coord, width, height(of Bounding Box) 
    //(compression type, compression level, precision of data, number of channels)
    

    /**
     * Constructor Simple
     */
    public MetaImageData() {
        state = ImageDataState.UNCOMPRESSED;
        compression_type = "XYZ";
    }
    /**
     * Constructor from ImageDataState
     */
    public MetaImageData(ImageDataState st) {
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
    /**
     * Get the compression type.
     */
    public String getCompressionType() {
    	return compression_type;
    }

    /**
     * Serialization of Object
     * @param out
     * @throws IOException
     */
    /*public void writeMetaObject(ObjectOutputStream out) throws IOException {
        //write normal fields
        //out.defaultWriteObject();
        out.writeObject(compression_type);
        out.writeObject(state);
    }*/

    /**
     * Serialization of Object
     * @param in
     * @throws IOException
     * @throws ClassNotFoundException
     */
    /*public void readMetaObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        //read normal fields
        //in.defaultReadObject();
        compression_type = (String) in.readObject();
        state = (ImageDataState) in.readObject();
        
    }*/

}
