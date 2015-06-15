# DeepManuscripts
Deep Learning on Historical Manuscripts


## Deep learning

### Dev Environment setup 
1. Install Eclipse (http://www.eclipse.org/downloads/)
* Install Maven (`sudo apt-get install maven`)
* In Eclipse, install Maven plugin
  * `Help -> Install new software -> Luna ->`
    search for *Maven*
  * select *m2e*, doesn't matter which one if you see many of them (e.g. in Collaboration)
* Clone the project from github
  (`git clone git@github.com:arvanito/DeepManuscripts.git`)
* Compile: in DeepManuscripts/deeplearning, do `mvn package`
* Import the project to Eclipse
  * `File -> Import -> Maven -> Existing Maven Projects -> Next`
  * Select *DeepManuscripts/deeplearning* as the root directory
  * `Finish`
* If you want to run the code locally, you also need to install Spark
  * https://spark.apache.org/downloads.html
  * you might want to add Spark's *bin* to $PATH

### Protocol buffers
Before proceeding, please read the basic tutorial:
https://developers.google.com/protocol-buffers/docs/javatutorial
It is written concisely.

.proto file contains the description/layout of your settings.
This is the file you need to modify if you want to add/remove settings.
The .proto file is compiled using the protoc tool which generates a .java file. 
The .java file contains all the classes generated from your .proto file definition.

1. Compiling the .proto file requires you to install the protoc packages. Please read the 
  'Compiling .proto files' section from the provided link
  * Note that you do not need to compile the .proto file if you do not want to add changes to it
  * If you modified the .proto file, for the changes to take effect run the command below in the deeplearning folder:
        `protoc -I=src/main/java --java_out=src/ src/main/java/deep_model_settings.proto`
    This will create a class in the src/main/java folder named DeepModelSettings.java
    You need to run `mvn package` again after you update DeepModelSettings.java
* Modyfing the .proto file 
  * The .proto file must be compiled after every change.
  * Every `message` is compiled into a Java class
  * A field of the message can be `optional`, `required` or `repeated`
  * Every field `field` of a message generates methods of the form 
     ** hasField() is true if the field is present (it is always true for required fields but can be false for optional and repeated)
     ** getField() / getFieldList() returns the actual field message / a list of messages for repeated fields
* Creating a .prototxt file //TODO
