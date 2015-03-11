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
