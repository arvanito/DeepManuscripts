
package main.java;

import java.io.IOException;

import org.apache.commons.io.FilenameUtils;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.mapred.lib.MultipleOutputFormat;
import org.apache.hadoop.mapreduce.security.TokenCache;
import org.apache.hadoop.util.Progressable;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

/**
 * FileOutputFormat class allowing to save a PairRDD 'imageFileName,imageRepresentation' to a folder, even with subdirectory structure
 */
public class ImageOutputFormat extends
        MultipleOutputFormat<String, ImageData> {

    @Override
    protected String generateFileNameForKeyValue(String key, ImageData value, String name) {
        return key.toString();
    }

    @Override
    protected String generateActualKey(String key, ImageData value) {
        return key;
    }

    public static class ImageRecordWriter implements
            RecordWriter<String, ImageData> {
        private final FSDataOutputStream outputStream;

        public ImageRecordWriter(FSDataOutputStream outputStream) {
            this.outputStream = outputStream;
        }

        @Override
        public void close(Reporter reporter) throws IOException {
            outputStream.close();
        }

        @Override
        public void write(String name, ImageData img)
                throws IOException {
            String ext = FilenameUtils.getExtension(name);
            MatOfByte buf = new MatOfByte();
            Highgui.imencode("."+ext, img.getImage(), buf);
            byte[] data  = buf.toArray();
            outputStream.write(data, 0, data.length);

        }
    }

    @Override
    protected RecordWriter<String, ImageData> getBaseRecordWriter(FileSystem fileSystem, JobConf jobConf, String s, Progressable progressable) throws IOException {
        Path codecClass1 = FileOutputFormat.getTaskOutputPath(jobConf, s);
        FileSystem codec1 = codecClass1.getFileSystem(jobConf);
        FSDataOutputStream stream = codec1.create(codecClass1, progressable);
        return new ImageRecordWriter(stream);
    }

    /**
     * Allows to overwrite/modify an existing folder
     */
    public void checkOutputSpecs(FileSystem ignored, JobConf job) throws FileAlreadyExistsException, InvalidJobConfException, IOException {
        Path outDir = getOutputPath(job);
        if(outDir == null && job.getNumReduceTasks() != 0) {
            throw new InvalidJobConfException("Output directory not set in JobConf.");
        } else {
            if(outDir != null) {
                FileSystem fs = outDir.getFileSystem(job);
                outDir = fs.makeQualified(outDir);
                setOutputPath(job, outDir);
                TokenCache.obtainTokensForNamenodes(job.getCredentials(), new Path[]{outDir}, job);
                //if(fs.exists(outDir)) {
                //    throw new FileAlreadyExistsException("Output directory " + outDir + " already exists");
                //}
            }

        }
    }
}