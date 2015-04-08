# Preprocessing #

## Setup (by Benoit, on MacOS 10.9) ##

Things are way simpler now!

In the preprocessing directory, just run `mvn package`, this will download from [a repository I created](https://github.com/Atanahel/opencv-maven-repo) the jar and native library of OpenCV. Only the versions for MacOS-64 and Linux-64 are available though.

The results are in `target` with the corresponding compiled jar, OpenCV jar is in `target/lib` and OpenCV native is in `target/native`.

## Installation test ##

### Running with spark-submit ###

After running the `mvn package`, you can use the jar directly :

1.  Have a folder `<input-folder>` in HDFS with some image files in it.
1.  Run `spark-submit --master <your-choice> --class main.java.TestMain target/DeepManuscriptPreprocessing-0.0.1.jar <input-folder> <output-folder>`.
1.  You should get some `part-*` files with the names of the images in the input folder and the number of pixels of each image.

The native library and the jar dependencies are packaged in the jar itself and get unpacked on each node.

### Running locally with the IDE ###

When you program on your machine and you want to be able to debug/test etc... :

1.  Import the maven project to your favorite IDE.
1.  Have a folder `<input-folder>` with some image files in it.
1.  Create a `Run` configuration based on `TestMain`.
1.  Set the VM argument to `-Djava.library.path=target/native`
1.  Set the program argument of the configuration to `--local <input-folder> <output-folder>`.
1.  Click Run.
1.  You should get some `part-*` files with the names of the images in the input folder and the number of pixels of each image.

The IDE should get the jar from maven but needs help to find the native library hence the VM parameter.

### OpenCV compilation (DEPRECATED NOW) ###

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