package main.java;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileAlreadyExistsException;
import org.apache.hadoop.mapred.InvalidJobConfException;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.lib.MultipleTextOutputFormat;
import org.apache.hadoop.mapreduce.security.TokenCache;

import java.io.IOException;

/**
 * OutputFormat class that outputs to a different files named with the key, and whose content is the value.
 */
public class MultiFileOutput extends MultipleTextOutputFormat<String, String> {
    @Override
    protected String generateFileNameForKeyValue(String key, String value, String name) {
        return key.toString();
    }

    @Override
    protected String generateActualKey(String key, String value) {
        return null;
    }

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
