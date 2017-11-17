package Task3

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object FpGrowth {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setAppName("market-basket-problem")
    conf.setMaster("local[2]")

    val sc = new SparkContext(conf)

    val fileRDD = sc.textFile("data sources/retail.xlsx")

    val products: RDD[Array[String]] = fileRDD.map(s => s.split(","))

    val fpg = new FPGrowth()
      .setMinSupport(0.2)

    val model = fpg.run(products)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    val minConfidence = 0.3
    val rules = model.generateAssociationRules(minConfidence)

    rules.collect().foreach { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]")
          + " => " + rule.consequent.mkString("[", ",", "]")
          + ", " + rule.confidence)
    }
  }
}
