package main.java;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

import java.io.Serializable;


/**
 * Representation of some image data.
 * Because this class will be used by Spark RDD, it needs to implement Serializable.
 * Two representations are kept, Compressed and Uncompressed. This should be transparent to external classes, but should allow
 * to drastically reduce drive and/or network bandwith if data is moved.
 */
public class ImageData implements Serializable {
    private static final long serialVersionUID = -151442211116649858L;

    //Uncompressed representation
    private Mat image;
    //Compressed representation
    private MatOfByte compressed_img;

    //Status of the representation
    public enum ImageDataState {UNCOMPRESSED, COMPRESSED, ERROR}
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
            boolean retValue = Highgui.imencode("png",image,compressed_img);
            if (retValue) {
                state = ImageDataState.ERROR;
            }else {
                state = ImageDataState.COMPRESSED;
            }
        }
    }

}