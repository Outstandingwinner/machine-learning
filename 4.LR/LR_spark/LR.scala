package spark.example

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.log4j.{Level,Logger}
import org.apache.spark.mllib.util.MLUtils


object LogisticRegression {
    val conf = new SparkConf().setMaster("local").setAppName("LogisticRegression")
    val sc = new SparkContext(conf)
    def main(args:Array[String]): Unit = {
        val data = MLUtils.loadLibSVMFile(sc, args(0))
        val splits = data.randomSplit(Array(0.6,0.4),seed = 11L)
        val parsedData =splits(0)
        val parsedTest =splits(1)

        val numiteartor = 50
        val model = LogisticRegressionWithSGD.train(parsedData,numiteartor) //训练模型

        val predictionAndLabels = parsedTest.map{                           //计算测试值
            case LabeledPoint(label,features) =>
            val prediction = model.predict(features)
                (prediction,label)                                              //存储测试值和预测值
        }
        predictionAndLabels.foreach(println)
        val trainErr = predictionAndLabels.filter( r => r._1 != r._2).count.toDouble / parsedTest.count
        println("trainErr： " +trainErr)
        val metrics = new MulticlassMetrics(predictionAndLabels)           //创建验证类
        val precision = metrics.precision                                   //计算验证值
        println("Precision= "+precision)
    }
}
