import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object kmeans {
  def main(args: Array[String]) {

    val conf = new SparkConf().setMaster("local[2]").setAppName("kmeans")
    val sc = new SparkContext(conf)

    val rawTrainingData = sc.textFile("hdfs://hadoop1:9000/Kmeans_train_set.data")
    val parsedData =
      rawTrainingData.filter(!isColumnNameLine(_)).map(line => {
        Vectors.dense(line.split(",").map(_.trim).filter(!"".equals(_)).map(_.toDouble))
      }).cache()

    // Cluster the data into two classes using KMeans

    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed=11L)
    val trainData = splits(0)
    val testData = splits(1)

    val numClusters = 8
    val numIterations = 30
    val runTimes = 3
    var clusterIndex: Int = 0
    val clusters: KMeansModel = KMeans.train(trainData, numClusters, numIterations, runTimes)

    println("Cluster Number:" + clusters.clusterCenters.length)

    println("Cluster Centers Information Overview:")
    clusters.clusterCenters.foreach(
      x => {
        println("Center Point of Cluster " + clusterIndex + ":")
        println(x)
        clusterIndex += 1
      })

    //begin to check which cluster each test data belongs to based on the clustering result
    testData.collect().foreach(testDataLine => {
      val predictedClusterIndex:
        Int = clusters.predict(testDataLine)
      println("The data " + testDataLine.toString + " belongs to cluster " + predictedClusterIndex)
    })

    println("Spark MLlib K-means clustering test finished.")

    val ecaluations=for(cluster<-Array(2,3,4,5,6,7,8)) yield {
      val clusterModule=KMeans.train(trainData,cluster,numIterations)
      val WSSSE=clusterModule.computeCost(trainData)
      (cluster,WSSSE)
    }
    ecaluations.sortBy(_._2).reverse.foreach(x => {println("clusterNum ="+x._1 + " WSSSE="+ x._2)})

  }

  private def isColumnNameLine(line: String): Boolean = {
    if (line != null && line.contains("Channel")) true
    else false
  }




}