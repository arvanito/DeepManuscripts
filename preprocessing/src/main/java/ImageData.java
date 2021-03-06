package main.java;

import org.opencv.core.Core;
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
 * to drastically reduce drive and/or network bandwith if data is moved.
 */
public class ImageData implements Serializable {
    private static final long serialVersionUID = -151442211116649858L;

    static {
        NativeLibraryLoader.load(Core.NATIVE_LIBRARY_NAME);
        NativeLibraryLoader.load("AndreaPipeline");
    }

    //Uncompressed representation
    transient private Mat image;
    //Compressed representation
    transient private MatOfByte compressed_img;

    //Status of the representation
    public enum ImageDataState {UNCOMPRESSED, COMPRESSED}
    private ImageDataState state;

    //Probably other image metadata needed (compression type, compression level, precision of data, number of channels)

    /**
     * Constructor from uncompressed data image.
     * @param image Image to be encapsulated. Data is NOT copied, only the reference.
     */
    public ImageData(Mat image) {
        this.image = image; //Could be image.clone() for safety.
        state = ImageDataState.UNCOMPRESSED;
    }

    /**
     * Constructor from compressed data image.
     * @param data Byte representation of a compressed image. Data is copied.
     */
    public ImageData(byte[] data) {
        this.compressed_img = new MatOfByte(data);
        state = ImageDataState.COMPRESSED;
    }

    /**
     * Get the uncompressed image. If the status was COMPRESSED, data is uncompressed.
     * @return a pointer to the uncompressed image.
     */
    public Mat getImage() {
        if (state == ImageDataState.COMPRESSED) {
            decompress();
        }
        return image;
    }

    /**
     * Get the compressed binary represenation of the image.
     * @return an array of the binary representation.
     */
    public byte[] getCompressedData() {
        if (state == ImageDataState.UNCOMPRESSED) {
            compress();
        }
        return compressed_img.toArray();
    }

    /**
     * Get the current compression status of the class.
     * @return current status.
     */
    public ImageDataState getStatus() {
        return state;
    }

    /**
     * Perform decompression and change the status, if needed.
     */
    public void decompress() {
        if (state == ImageDataState.COMPRESSED) {
            image = Highgui.imdecode(compressed_img, Highgui.IMREAD_GRAYSCALE);
            state = ImageDataState.UNCOMPRESSED;
        }
    }

    /**
     * Perform compression and change the status, if needed.
     */
    public void compress() {
        if (state == ImageDataState.UNCOMPRESSED) {
            if (compressed_img==null)
                compressed_img=new MatOfByte();
            boolean retValue = Highgui.imencode(".png",image,compressed_img);
            state = ImageDataState.COMPRESSED;
        }
    }

    /**
     * Serialization of Object
     * @param out
     * @throws IOException
     */
    private void writeObject(ObjectOutputStream out) throws IOException {
        //compress if not
        compress();
        //write normal fields
        out.defaultWriteObject();
        //write compressed data
        byte[] data = compressed_img.toArray();
        out.writeInt(data.length);
        out.write(data);
    }

    /**
     * Serialization of Object
     * @param in
     * @throws IOException
     * @throws ClassNotFoundException
     */
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        //read normal fields
        in.defaultReadObject();
        //read compressed data
        int size = in.readInt();
        System.out.println(size);
        byte[] data = new byte[size];
        int toBeRead=size;
        while (toBeRead>0) {
            int nRead = in.read(data, size-toBeRead, toBeRead);
            if(nRead<0)
                throw new IOException();
            toBeRead-=nRead;
        }
        compressed_img = new MatOfByte(data);
    }

}