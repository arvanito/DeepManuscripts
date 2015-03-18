# Preprocessing #

## Setup (by Benoit, on MacOS 10.9) ##

### OpenCV ###

Because we need image processing tools, OpenCV is a very good choice. However we can not just add the library to the `pom.xml` of Maven (or if someone can find a way... tell me).

You need a working installation of OpenCV with Java support included. See [there](http://docs.opencv.org/doc/tutorials/introduction/desktop_java/java_dev_intro.html).

For Windows, you can probably just download it.
For Linux, there is probably a way with the package manager??
For Mac, you can probably either use brew : `brew install opencv --with-java` or Macports `sudo port install opencv +java`. I compiled it myself.

If you want to compile it yourself on Mac/Linux, you first need `cmake` and `ant` installed. Check the link above but roughly :

1. `git clone git://github.com/Itseez/opencv.git` checkout source
1. `mkdir opencv-build` create separate building directory
1. `cd opencv-build/`
1. `cmake -DBUILD_SHARED_LIBS=OFF -G "Unix Makefiles" ../opencv` create makefile
1. `ccmake .` If you want to disable some things like tests or perf evals. The video module usually cause some problems on Mac as well.
1. `make -j8` to parallelize making process with 8 threads.

You should have the jar as something like `opencv-build/bin/opencv-2411.jar` and the native library in `opencv-build/lib`.

### Installation test ###

(Refer to the detailled guide by Artu for project creation, I use IntelliJ but it should not matter)

1.   Locate the `<path-to-opencv-jar-file>` and the `<directory-containing-opencv-jni-libs>`.
1.   In your project configuration, add an external library. You need to set the jar file and the directory containing the native library.
1.   Have a folder `<input-folder>` with some image files in it.
1.  Run the example, two possible ways :
    * From the IDE only for local and testing purpose.
        1.  Create a `Run` configuration based on `TestMain`.
        1.  Set the arguments of the configuration to `--local <input-folder> <output-folder>`.
        1.  Click Run.
    * With `spark-submit`
        1.   Compile the files to `<your-jar-file>`.
        1.   Run `spark-submit --master local --class main.java.TestMain --jars <path-to-opencv-jar-file> --driver-library-path <directory-containing-opencv-jni-libs> <your-jar-file> <input-folder> <output-folder>`.
1.   You should get some `part-*` files with the names of the images in the input folder and the number of pixels of each image.