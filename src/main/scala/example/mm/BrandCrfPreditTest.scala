
import com.intel.imllib.crf.nlp.{CRFModel, Sequence, VerboseLevel1}
import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, Path}
import org.apache.hadoop.util.GenericOptionsParser
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

//用于测试线上代码---BrandCrfPredit
object BrandCrfPreditTest {

  /**
    * @param poiId wdcid
    * @param poiName wdcname
    * @param crfFeaturePoiName 经过crf标记后的结果
    * @param suspectedBrandName 提取品牌名称
    * @param suspectedBrandType 提取品牌名称方式 0表示默认值 1表示crf提取
    */
  case class Poi(poiId:Long,poiName:String,crfFeaturePoiName:String,suspectedBrandName:String,suspectedBrandType:Int)


  import org.apache.spark.sql.functions._
  import scala.collection.JavaConversions._
  //crf格式转换
  val crfFeaturePoiname = udf((poiName:String) => {
    //格式 O|--|卤  O|--|巧  O|--|儿  O|--|（  O|--|黄  O|--|石  O|--|北  O|--|站  O|--|店  O|--|）
    //默认预测时,不需要提供label,因此设置默认值O。
    CrfTermConvert.predict(poiName).map("O|--|"+_).mkString("\t")
  })

  def invoke(args: Array[String],appName:String): Unit = {

    val conf = new SparkConf().setAppName(appName).setMaster("local")
    val sc = new SparkContext(conf)
    val argConf = new Configuration()
    val parser = new GenericOptionsParser(argConf, args)
    val out_table = argConf.get("output_table")
    val remainArgs = parser.getRemainingArgs
    val sqlContext = new SQLContext(sc)


    val filepath1 = "/Users/maming/Downloads/test/brand_dig_crf_feature_test.txt"
    val df = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true").option("inferSchema", "false").option("delimiter", "\t").load(filepath1)
    df.show(100)

    val brandPoolDf = df
      .withColumn("crf_feature_poiname", col = crfFeaturePoiname(col("wdc_name")))
      .cache()

    brandPoolDf.show(100)

    val brandPoolResultDf = brandPoolDf.rdd.map(row =>
      Poi(StringUtil.toLong(row.getAs[String]("id").toLong),
        StringUtil.trim(row.getAs[String]("name")),
        StringUtil.trim(row.getAs[String]("crf_feature_poiname")),
        "", 0)
    ).filter(_.crfFeaturePoiName.nonEmpty)
      .mapPartitions(iter => {

        //加载model
        val modelPath: Path = new Path("/Users/maming/Desktop/mm/dev/crf/test/spark_version/spark_crf_model")
        val input: FSDataInputStream = modelPath.getFileSystem(new Configuration()).open(modelPath)
        val model = CRFModel.loadArray(input)
          .setNBest(10)
          .setVerboseMode(VerboseLevel1)
          .setFeatureIndex()

        iter.map(poi => {
          //模型预测疑似品牌名
          val sequence: Sequence = model.predict(Sequence.deSerializer(poi.crfFeaturePoiName)) //预测返回新的sequence

          val tempName = sequence.sequence // token集合
            .filter(token => !token.label.equals("O")) //过滤无效的label
            .map(token => token.tags(0))
            .map( //英文和数字时加入空格,并且结果trim
              _ match {
                case term: String
                  if (!term.matches("[^\\\\u4e00-\\\\u9fa5\\\\u0030-\\\\u0039\\\\u0041-\\\\u005a\\\\u0061-\\\\u007a]")) => term + " "
                case term: String => term
              })
            .mkString("").trim

          val (suspectedBrandName, suspectedBrandType) = tempName match {
            case tempName: String if (StringUtils.isBlank(tempName)) => (PoiNameConvert.format(poi.poiName).trim, 0)
            case tempName: String => (tempName, 1)
          }
          poi.copy(crfFeaturePoiName = sequence.toString,suspectedBrandName = suspectedBrandName,suspectedBrandType = suspectedBrandType)
        })
      })
    brandPoolResultDf.take(100).foreach(println(_))
  }

  def main(args: Array[String]): Unit = {

    invoke(args,"BrandCrfPredit")

  }

}
