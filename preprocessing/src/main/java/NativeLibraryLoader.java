package main.java;


import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * NativeLibraryLoader class that load native library from the system path. If this failed, it assumes the native library
 * is a ressource file in the Jar, unpacks it and manually loads it.
 * Heavily inspired from https://github.com/PatternConsulting/opencv/blob/stable/src/main/java/nu/pattern/OpenCV.java
 * No deletion of the temporary directory is done at the end.
 */
public class NativeLibraryLoader {
    static Path destinationFolder;

    static {
        try {
            destinationFolder = Files.createTempDirectory("");
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void load(final String libBaseName) {
        try {
            System.loadLibrary(libBaseName);
            System.out.println(libBaseName+" found in java.library.path");
        }catch (UnsatisfiedLinkError e) {
            System.out.println(libBaseName + " not found in java.library.path");
            System.out.println("Trying to extract from jar...");
            final String fullLibName = System.mapLibraryName(libBaseName);
            final String location = "/"+fullLibName;
            final InputStream binary = NativeLibraryLoader.class.getResourceAsStream(location);

            try {
                final Path destination = destinationFolder.resolve(fullLibName);
                Files.copy(binary, destination);
                System.out.println("Copying "+fullLibName+" to "+destination);
                System.load(destination.normalize().toString());
                System.out.println("SUCCESS Loading!");
            }catch (Exception e2) {
                System.out.println("FAILURE");
                e2.printStackTrace();
            }
        }
    }

}
