# DeepManuscripts
Deep Learning on Historical Manuscripts


## Deep learning

### Dev Environment setup (by Arttu, for Ubuntu 14.10 & Eclipse Luna)
Disclaimer: this is approximately how I set this up.
Your mileage might wary. Also, it should be totally possible to develop
without Eclipse, but I find it quite useful.

1. install Eclipse (http://www.eclipse.org/downloads/)
* install Maven (`sudo apt-get install maven`)
* in Eclipse, install Maven plugin
  * `Help -> Install new software -> Luna ->`
    search for *Maven*
  * select *m2e*, doesn't matter which one if you see many of them (e.g. in Collaboration)
* clone the project from github
  (`git clone git@github.com:arvanito/DeepManuscripts.git`)
* compile: in DeepManuscripts/deeplearning, do `mvn package`
* import the project to Eclipse
  * `File -> Import -> Maven -> Existing Maven Projects -> Next`
  * select *DeepManuscripts/deeplearning* as the root directory
  * `Finish`
* if you want to run the code locally, you also need to install Spark
  * https://spark.apache.org/downloads.html
  * you might want to add Spark's *bin* to $PATH
* to test if everything works:
  * create a file called `test_in.txt`, with content `123 456 789`
    (tabs, not spaces)
  * in *DeepManuscripts/deeplearning*, do (supposing Spark's *bin* is in $PATH)
    `spark-submit --class main.java.DeepLearningMain --master local[1]
    target/DeepManuscriptLearning-0.0.1.jar test_in.txt test_out`
  * it should execute, and you should end up with folder `test_out` with
    file `part-00000` containing `[123.0,456.0,789.0]`

### Protocol buffers
Before proceeding, please read the basic tutorial:
https://developers.google.com/protocol-buffers/docs/javatutorial
It is written concisely.

.proto file contains the description/layout of your settings.
This is the file you need to modify if you want to add/remove settings.
The .proto file is compiled using the protoc tool which generates a .java file. 
The .java file contains all the classes generated from your .proto file definition.

* Compiling the .proto file requires you to install the protoc packages. Please read the 
  'Compiling .proto files' section from the provided link
  * Note that you do not need to compile the .proto file if you do not want to add changes to it
  * If you modified the .proto file, for the changes to take effect run the command below in the deeplearning folder:
         protoc -I=src/main/java --java_out=src/ $src/main/java/deep_model_settings.proto
    This will create a class in the src/main/java folder named DeepModelSettings.java

* Modyfing the .proto file //TODO 
* Creating a .prototxt file //TODO
* Available methods //TODO

 
