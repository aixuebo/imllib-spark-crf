
import com.intel.imllib.crf.nlp._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * CRF-spark版本 模型的使用demo
  * 介绍了如何使用模型训练数据、预测数据、测试准确率
  */
object CRFExample {

  val templateFile = "/Users/maming/Desktop/mm/dev/crf/test/spark_version/template_file_spark"
  val trainFile = "/Users/maming/Desktop/mm/dev/crf/test/spark_version/local/train_20_spark"
  val testFile = "/Users/maming/Desktop/mm/dev/crf/test/spark_version/local/test_20_spark"
  val modelFile = "/Users/maming/Desktop/mm/dev/crf/test/spark_version/local/test_model" //输出模型

  //训练
  def train(): Unit ={

    val conf = new SparkConf().setMaster("local").setAppName("CRFExample")
    val sc = new SparkContext(conf)

    val templates: Array[String] = scala.io.Source.fromFile(templateFile).getLines().filter(_.nonEmpty).toArray
    val trainRDD: RDD[Sequence] = sc.textFile(trainFile).filter(_.nonEmpty).map(Sequence.deSerializer)

    //模型训练
    val model: CRFModel = CRF.train(templates, trainRDD, 0.25, 1, 100, 1E-3, "L1")

    //输出模型
    val modelPath : Path = new Path(modelFile)
    val output : FSDataOutputStream = modelPath.getFileSystem(new Configuration()).create(modelPath)
    CRFModel.saveArray(model,output)

    sc.stop()
  }

  //预测
  def predic(): Unit ={
    val conf = new SparkConf().setMaster("local").setAppName("CRFExample")
    val sc = new SparkContext(conf)
    val testRDD: RDD[Sequence] = sc.textFile(testFile).filter(_.nonEmpty).map(Sequence.deSerializer)

    //加载模型
    val modelPath : Path = new Path(modelFile)
    val input : FSDataInputStream = modelPath.getFileSystem(new Configuration()).open(modelPath)
    val modelNew = CRFModel.loadArray(input)

    //预测
    val results: RDD[Sequence] = modelNew.setNBest(10)
      .setVerboseMode(VerboseLevel1)
      .predict(testRDD)

    //预测结果与原始集做校验
    val score = results
      .zipWithIndex()
      .map(_.swap)
      .join(testRDD.zipWithIndex().map(_.swap)) //预测结果与test进行join,相同序号的做join
      .map(_._2) //获取预测结果和test内容
      .map(x => x._1.compare(x._2)) //预测结果相同
      .reduce(_ + _)
    val total = testRDD.map(_.toArray.length).reduce(_ + _)


    results
      .zipWithIndex()
      .map(_.swap)
      .join(testRDD.zipWithIndex().map(_.swap)) //预测结果与test进行join,相同序号的做join
      .map(_._2) //获取预测结果和test内容
      .map(x => x._1.sequence).map(tokens =>{
      tokens.mkString("\t")
    }).foreach(println(_))

    println(s"Prediction Accuracy: $score / $total")
    sc.stop()
  }

  def main(args: Array[String]) {
    if (args.length != 3) {
      println("CRFExample <templateFile> <trainFile> <testFile>")
    }

    train()
    predic()

  }
}

