package com.badou

import scala.math._
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object cf {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("CF test")
    val sc = new SparkContext(conf)
    val lines = sc.textFile(args(0))
    val output_path = args(1).toString

    val topn = 10
    val max_prefs_per_user = 20
    val score_thd = 0.001

    // step1. Obtain New UI Matrix
    val ui_rdd = lines.map { x =>
      val fields = x.split("\t")
      (fields(0).toString, (fields(1).toString, fields(2).toDouble))
    }.filter{ x=>
      x._2._2 > score_thd
    }.groupByKey().flatMap{ x =>
      val user = x._1
      val is_list = x._2
      val is_arr = is_list.toArray
      var is_arr_len = is_arr.length
      if (is_arr_len > max_prefs_per_user) {
        is_arr_len = max_prefs_per_user
      }
      var i_us_arr = ArrayBuffer[(String, (String, Double))]()
      for (i <- 0 until is_arr_len) {
        i_us_arr += ((is_arr(i)._1, (user, is_arr(i)._2)))
      }
      i_us_arr
    }.groupByKey().flatMap { x =>
      val item = x._1
      val u_list = x._2
      val us_arr = u_list.toArray
      var sum: Double = 0
      for (i <- 0 until us_arr.length) {
        sum += pow(us_arr(i)._2, 2)
      }
      sum = sqrt(sum)

      var u_is_arr = ArrayBuffer[(String, (String, Double))]()
      for (i <- 0 until us_arr.length) {
        u_is_arr += ((us_arr(i)._1, (item, us_arr(i)._2 / sum)))
      }
      u_is_arr
    }.groupByKey()

    // step2. gen pairs and sum and output
    val ii_rdd = ui_rdd.flatMap { x=>
      val is_arr = x._2.toArray
      var ii_s_arr = ArrayBuffer[((String, String), Double)]()
      for (i <- 0 until is_arr.length) {
        for (j <- i + 1 until is_arr.length) {
          ii_s_arr += (((is_arr(i)._1, is_arr(j)._1), is_arr(i)._2 * is_arr(j)._2))
        }
      }
      ii_s_arr
    }.groupByKey().map { x =>
      val ii_pair = x._1
      val s_list = x._2
      val s_arr = s_list.toArray
      val len = s_arr.length
      var score:Double = 0.0
      for (i <- 0 until len) {
        score += s_arr(i)
      }
      (ii_pair._1, (ii_pair._2, score))
    }.flatMap { x =>
      var arr = ArrayBuffer[(String, (String, Double))]()
      arr += ((x._1, (x._2._1, x._2._2)))
      arr += ((x._2._1, (x._1, x._2._2)))
      arr
    }.groupByKey().map { x =>
      val main_item = x._1
      val bs_list = x._2
      val bs_arr = bs_list.toArray.sortWith(_._2 > _._2)
      var len = bs_arr.length
      if (len > topn) {
        len = topn
      }

      val s = new StringBuilder
      for (i <- 0 until len) {
        val item = bs_arr(i)._1
        val score = "%1.6f" format bs_arr(i)._2
        s.append(item + ":"+ score)
        if (i !=(len -1)) {
          s.append(",")
        }
      }
      main_item + "\t" + s
    }.saveAsTextFile(output_path)
  }
}
