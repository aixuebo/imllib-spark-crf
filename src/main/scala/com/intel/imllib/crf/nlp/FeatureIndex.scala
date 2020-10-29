/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.imllib.crf.nlp

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{DenseVector => BDV}

import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast

private[nlp] class FeatureIndex extends Serializable {

  var maxID = 0
  var alpha: BDV[Double] = _
  var tokensSize = 0 //tag特此长度
  var labels: Array[String] = _ //过滤重复后的label集合

  //模板语法
  val unigramTempls = new ArrayBuffer[String]()
  val bigramTempls = new ArrayBuffer[String]()

  val dic = mutable.HashMap[String, (Int, Int)]()
  val kMaxContextSize = 4
  val BOS = Array("_B-1", "_B-2", "_B-3", "_B-4")
  val EOS = Array("_B+1", "_B+2", "_B+3", "_B+4")

  def initAlpha() = {
    alpha = BDV.zeros[Double](maxID)
    alpha
  }
  
  def openTagSet(sentence: Sequence) = {
    val tokenNum = sentence.toArray.map(_.tags.length).distinct //除了label外,tag的特征长度,要求长度是相同的
    require(tokenNum.length == 1,
      "The number of columns should be fixed in each token!")

    labels = sentence.toArray.map(_.label) //该句子的所有label集合
    tokensSize = tokenNum.head
    (labels, tokensSize) //该句子的所有label集合 以及 tag特此长度
  }

  /**
   * Build feature index
   */
  def buildFeatures(tagger: Tagger): Tagger = {
    List(unigramTempls, bigramTempls).foreach{ templs => //循环每一种模板集合
      tagger.x.foreach { token => //循环每一个token
        if (tagger.x.head != token || templs.head.head.equals('U')) {
          tagger.featureCacheIndex.append(tagger.featureCache.length)
          templs.foreach { templ => //循环每一个模板
            val os = applyRule(templ, tagger.x.indexOf(token), tagger)
            val id = dic.getOrElse(os, (-1, 0))._1
            if (id != -1) tagger.featureCache.append(id)
          }
          tagger.featureCache.append(-1)
        }
      }
    }
    tagger
  }

  def buildDictionary(tagger: Tagger) = {
    val dicLocal = mutable.HashMap[String, Int]()
    List(unigramTempls, bigramTempls).foreach{ templs =>
      tagger.x.foreach{ token =>
        if(tagger.x.head != token || templs.head.head.equals('U'))
          templs.foreach{ templ =>
            val os = applyRule(templ, tagger.x.indexOf(token), tagger)
            if (dicLocal.get(os).isEmpty)
              dicLocal.update(os, 1)
            else {
              val idx = dicLocal.get(os).get + 1
              dicLocal.update(os, idx)
              }
            }
      }
    }
    dicLocal
  }

  //模板 U13:%x[-1,0]/%x[0,0] 或者 U12:%x[6,0]
  def applyRule(src: String, idx: Int, tagger: Tagger): String = {
    val templ = src.split(":")
    if (templ.size == 2) {
      val cols = templ(1).split("/").map(_.substring(2))
      templ(0) + ":" + cols.map(getIndex(_, idx, tagger)).reduce(_ + "/" + _)
    } else if (templ.size == 1) {
      templ(0)
    } else
        throw new RuntimeException("Incompatible formats in Template")
  }

  def getIndex(src: String, pos: Int, tagger: Tagger): String = {
    //因为模板是U12:%x[6,0],此时src是[6,0],因此要去除[]中括号
    val coor = src.drop(1).dropRight(1).split(",")
    require(coor.size == 2, "Incompatible formats in Template")
    val row = coor(0).toInt
    val col = coor(1).toInt
    if (row < -kMaxContextSize || row > kMaxContextSize ||
      col < 0 || col >= tokensSize) {
      throw new RuntimeException("Incompatible formats in Template")
    }
    val idx = pos + row
    if (idx < 0) {
      BOS(- idx - 1)
    } else if (idx >= tagger.x.size) {
      EOS(idx - tagger.x.size)
    } else {
      tagger.x(idx)(col)
    }
  }

  /**
    * Read one template file
    *
    * @param lines the template file
    * 读取模板文件,向集合中添加模板信息
    */
  def openTemplate(lines: Array[String]): Unit = {
    var i: Int = 0
    lines.foreach { t =>
      t.head match{
        case 'U' => unigramTempls += t
        case 'B' => bigramTempls += t
        case '#' =>
        case _ => throw new RuntimeException("Incompatible formats in Templates")
      }}
  }

  def saveModel: CRFModel = {
    val head = new ArrayBuffer[String]()

    head.append("maxid:")
    head.append(maxID.toString)
    head.append("cost-factor:")
    head.append(1.0.toString)
    head.append("xsize:")
    head.append(tokensSize.toString)
    head.append("Labels:")
    labels.foreach(head.append(_))
    head.append("UGrams:")
    unigramTempls.foreach(head.append(_))
    head.append("BGrams:")
    bigramTempls.foreach(head.append(_))

    CRFModel(head.toArray, dic.map { case (k, v) => (k, v._1) }.toArray, alpha.toArray)
  }

  def readModel(models: CRFModel) = {
    val contents: Array[String] = models.head
    val labelsBuffer = new ArrayBuffer[String]()
    models.dic.foreach{case(k, v) => dic.update(k, (v, 1))}
    alpha = new BDV(models.alpha)

    var i: Int = 0
    var readMaxId: Boolean = false
    var readCostFactor: Boolean = false
    var readXSize: Boolean = false
    var readLabels: Boolean = false
    var readUGrams: Boolean = false
    var readBGrams: Boolean = false
    val alpha_tmp = new ArrayBuffer[Double]()
    while (i < contents.length) {
      contents(i) match {
        case "maxid:" =>
          readMaxId = true
        case "cost-factor:" =>
          readMaxId = false
          readCostFactor = true
        case "xsize:" =>
          readCostFactor = false
          readXSize = true
        case "Labels:" =>
          readXSize = false
          readLabels = true
        case "UGrams:" =>
          readLabels = false
          readUGrams = true
        case "BGrams:" =>
          readUGrams = false
          readBGrams = true
        case _ =>
          i -= 1
      }
      i += 1
      if (readMaxId) {
        maxID = contents(i).toInt
      } else if (readXSize) {
        tokensSize = contents(i).toInt
      } else if (readLabels) {
        labelsBuffer.append(contents(i))
      } else if (readUGrams) {
        unigramTempls.append(contents(i))
      } else if (readBGrams) {
        bigramTempls.append(contents(i))
      }
      i += 1
    }
    labels = labelsBuffer.toArray
    this
  }

  //将训练和测试集合RDD进行处理
  def openTagSetDist(trains: RDD[Sequence]) {
    val features: RDD[(Array[String], Int)] = trains.map(openTagSet) //该句子的所有label集合 以及 tag特此长度
    val tokensSizeCollect = features.map(_._2).distinct().collect() //要求tag特此长度是唯一的
    require(tokensSizeCollect.length == 1,
      "The number of columns should be fixed in each token!")
    tokensSize = tokensSizeCollect.head
    labels = features.map(_._1.distinct).flatMap(l => l).distinct().collect() //过滤重复后的label集合,数据量没多少,因此可以缓存到本地
  }

  def buildDictionaryDist(taggers: RDD[Tagger],  bcFeatureIdxI: Broadcast[FeatureIndex], freq: Int) {
    //filter : use features that occur no less than freq(default 1)
    val dictionary = taggers.flatMap(tagger => {
      bcFeatureIdxI.value.buildDictionary(tagger)
    }).reduceByKey(_ + _)
      .filter(_._2 >= freq)
    val dictionaryUni: RDD[(String, (Int, Int))] = dictionary.filter(_._1.head == 'U').zipWithIndex()
      .map{ case((feature, frequency), featureID) =>
        (feature, (featureID.toInt * bcFeatureIdxI.value.labels.size, frequency))
      }
    val bcOffSet = taggers.context.broadcast(dictionaryUni.count().toInt * labels.size)
    val dictionaryBi: RDD[(String, (Int, Int))] = dictionary.filter(_._1.head == 'B').zipWithIndex()
      .map{ case((feature, frequency), featureID) =>
        (feature, (featureID.toInt * bcFeatureIdxI.value.labels.size * bcFeatureIdxI.value.labels.size + bcOffSet.value, frequency))
      }

    //TODO local模式,似乎会消耗性能,有待优化
    val dictionaryGram = dictionaryUni.union(dictionaryBi).collect()

    dictionaryGram.foreach{case(k, v) => dic.update(k, v)}
    maxID = dictionaryGram.map(_._2._1).max + labels.size * labels.size

  }
}
