
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext,SparkConf}
import org.apache.spark.{SparkContext,SparkConf}

object naiveBayes {
  def main(args: Array[String]) {


    val conf = new SparkConf().setAppName("naiveBayes").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.textFile("hdfs://hadoop1:9000/a.txt")
    val parsedData =data.map { line =>
      val parts =line.split(',')
      LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    val splits = parsedData.randomSplit(Array(0.6,0.4),seed = 11L)
    val training =splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(training,lambda = 1.0)

    val predictionAndLabel= test.map(p => (model.predict(p.features),p.label))
    val accuracy =1.0 *predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println("accuracy-->"+accuracy)
    println("Predictionof (0.0, 2.0, 0.0, 1.0):"+model.predict(Vectors.dense(0.0,2.0,0.0,1.0)))
    println("Predictionof (2.0, 1.0, 0.0, 0.0):"+model.predict(Vectors.dense(0.0,2.0,0.0,1.0)))
  }
}
