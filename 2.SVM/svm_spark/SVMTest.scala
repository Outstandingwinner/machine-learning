import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.util._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.classification.{ SVMModel, SVMWithSGD }
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object SVMTest {
  def main(args: Array[String]) {


    val conf = new SparkConf().setAppName("svm").setMaster("local")
    val sc = new SparkContext(conf)

    val examples = MLUtils.loadLibSVMFile(sc, "hdfs://hadoop1:9000/svm_train.data")
    val splits = examples.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val numTraining = training.count()
    val numTest = test.count()
    println(s"Training: $numTraining, test: $numTest.")

    val numIterations = 1000
    val stepSize = 0.5
    val miniBatchFraction = 2.0

    val model = SVMWithSGD.train(training, numIterations, stepSize, miniBatchFraction)

    val prediction = model.predict(test.map(_.features))
    val predictionAndLabel = prediction.zip(test.map(_.label))

    val metrics = new MulticlassMetrics(predictionAndLabel)

    val precision = metrics.precision
    println("Precision = " + precision)

  }
}
