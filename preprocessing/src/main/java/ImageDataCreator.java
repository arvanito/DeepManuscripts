package main.java;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

/**
 * Created by benoit on 12/03/2015.
 */
public class ImageDataCreator implements Function<PortableDataStream, ImageData> {
    @Override
    public ImageData call(PortableDataStream portableDataStream) throws Exception {
        return new ImageData(portableDataStream.toArray());
    }
}