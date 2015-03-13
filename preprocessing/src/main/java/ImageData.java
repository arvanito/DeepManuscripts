package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

import java.io.Serializable;



public class ImageData implements Serializable {
    private static final long serialVersionUID = -151442211116649858L;

    private Mat image;
    private MatOfByte compressed_img;

    public enum ImageDataState {UNCOMPRESSED, COMPRESSED, ERROR}
    private ImageDataState state;

    public ImageData(Mat image) {
        this.image = image;
        state = ImageDataState.UNCOMPRESSED;
    }

    public ImageData(byte[] data) {
        this.compressed_img = new MatOfByte(data);
        state = ImageDataState.COMPRESSED;
    }

    public Mat getImage() {
        if (state == ImageDataState.COMPRESSED) {
            decompress();
        }
        return image;
    }

    public ImageDataState getStatus() {
        return state;
    }

    public void decompress() {
        if (state == ImageDataState.COMPRESSED) {
            image = Highgui.imdecode(new MatOfByte(compressed_img), Highgui.IMREAD_GRAYSCALE);
            state = ImageDataState.UNCOMPRESSED;
        }
    }

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