package main.java;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.lib.MultipleTextOutputFormat;

/**
 * OutputFormat class that outputs to a different files named with the key, and whose content is the value.
 */
public class MultiFileOutput extends MultipleTextOutputFormat<Text, Text> {
    @Override
    protected String generateFileNameForKeyValue(Text key, Text value, String name) {
        return key.toString();
    }

    @Override
    protected Text generateActualKey(Text key, Text value) {
        return null;
    }
}
