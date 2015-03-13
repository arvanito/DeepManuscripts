# Preprocessing
Deep Learning on Historical Manuscripts


## Setup (by Benoit, on MacOS 10.9)

### OpenCV

Because we need image processing tools, OpenCV is a very good choice. However we can not just add the library to the `pom.xml` of Maven (or if someone can find a way... tell me).
You need a working installation of OpenCV with Java support included (Windows users can just download it, Linux/MacOS can probably use a package manager, I compiled it myself).

### Installation test

(Refer to the detailled guide by Artu for project creation, I use IntelliJ but it should not matter)
1. Compile the files to a `<your-jar-file>`.
* Have a folder `<input-folder>` with some image files in it.
* Locate the `<path-to-opencv-jar-file>` and the `<directory-containing-opencv-jni-libs>`.
* Run `spark-submit --master local --class main.java.TestMain --jars <path-to-opencv-jar-file> --driver-library-path <directory-containing-opencv-jni-libs> <your-jar-file> <input-folder> <output-folder>`.
* You should get some `part-00000` file with the name of the images in the input folder and the number of pixels of each image.