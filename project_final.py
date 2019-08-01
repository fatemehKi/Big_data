# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:08:28 2019

@author: fkiai
"""

from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id
from functools import reduce
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from pyspark.sql import Row
#from pyspark.mllib.feature import HashingTF, IDF
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors

SpSession = SparkSession.builder.master("local").appName("py_spark").getOrCreate()
SpContext = SpSession.sparkContext


datalines = SpContext.textFile("sparkml/Comment_Classification_WOH.csv")

"""----------------------------------------------------------------------------
Cleanup Data
----------------------------------------------------------------------------"""
parts=datalines.map(lambda l:l.split(','))
cmnt2=parts.map(lambda p:Row(Comment=p[0]))
cmnt3=parts.map(lambda p:Row(HATE=int(p[2]), INSULT=int(p[3]), OBSENCE=int(p[4]), SEVER_TOXIC=int(p[6]), THREAD=int(p[7]), TOXIC=int(p[8])))

dataset2=SpSession.createDataFrame(cmnt2) ##dataset for the comments
dataset3=SpSession.createDataFrame(cmnt3)

Comments_cl = dataset2.select("Comment").rdd.flatMap(lambda x: x)

############################# Lower Case
lowerCase_sentRDD=Comments_cl

############################# Tokenization
def sent_TokenizeFunct(x):
    return nltk.sent_tokenize(x)
sentenceTokenizeRDD = lowerCase_sentRDD.map(sent_TokenizeFunct)


def word_TokenizeFunct(x):
    splitted = [word for line in x for word in line.split()]
    return splitted
wordTokenizeRDD = sentenceTokenizeRDD.map(word_TokenizeFunct)

############################# Removing Stop Words
def removeStopWordsFunct(x):
    #from nltk.corpus import stopwords
    stop_words=set(stopwords.words('english'))
    filteredSentence = [w for w in x if not w in stop_words]
    return filteredSentence
stopwordRDD = wordTokenizeRDD.map(removeStopWordsFunct)

############################# Removing Punctuations
def removePunctuationsFunct(x):
    list_punct=list(string.punctuation)
    filtered = [''.join(c for c in s if c not in list_punct) for s in x]
    filtered_space = [s for s in filtered if s] #remove empty space
    return filtered_space
rmvPunctRDD = stopwordRDD.map(removePunctuationsFunct)

############################# Leminization
def lemmatizationFunct(x):
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    finalLem = [lemmatizer.lemmatize(s) for s in x]
    return finalLem
lem_wordsRDD = rmvPunctRDD.map(lemmatizationFunct)


############################ joining tokens.. not required.
def joinTokensFunct(x):
    #joinedTokens_list = []
    x = " ".join(x)
    return x
joinedTokens = lem_wordsRDD.map(joinTokensFunct)
df2 = joinedTokens.map(lambda x: (x, )).toDF()

###name assignment
df3 = df2.selectExpr("_1 as Comment")
df3.printSchema()


############################ TF-IDF
df_s = lem_wordsRDD.map(lambda x: (x, )).toDF()
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(df_s.columns[1])

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

"""--------------------------------------------------------------------------
Perform Data Analytics
-------------------------------------------------------------------------"""
############################ Correlation analysis
for i in dataset3.columns:
    if not( isinstance(dataset3.select(i).take(1)[0][0], int)):
        print( "Correlation to toxic for ", i, dataset3.stat.corr('toxic',i))



############################ Transform to a Data Frame for input to Machine Learing
def transformToLabeledPoint1(row):
    lp = ( row["out1"], \
                Vectors.dense([row["Comment"],\
                        row["out2"], \
                        row["out3"], \
                        row["out4"],
                        row["out5"],
                        row["out6"]]))
    return lp

def transformToLabeledPoint2(row):
    lp = ( row["out2"], \
                Vectors.dense([row["Comment"],\
                        row["out1"], \
                        row["out3"], \
                        row["out4"],
                        row["out5"],
                        row["out6"]]))
    return lp

def transformToLabeledPoint3(row):
    lp = ( row["out3"], \
                Vectors.dense([row["Comment"],\
                        row["out1"], \
                        row["out2"], \
                        row["out4"],
                        row["out5"],
                        row["out6"]]))
    return lp



def transformToLabeledPoint4(row):
    lp = ( row["out4"], \
                Vectors.dense([row["Comment"],\
                        row["out1"], \
                        row["out2"], \
                        row["out3"],
                        row["out5"],
                        row["out6"]]))
    return lp

def transformToLabeledPoint5(row):
    lp = ( row["out5"], \
                Vectors.dense([row["Comment"],\
                        row["out1"], \
                        row["out2"], \
                        row["out3"],
                        row["out4"],
                        row["out6"]]))
    return lp

def transformToLabeledPoint6(row):
    lp = ( row["out6"], \
                Vectors.dense([row["Comment"],\
                        row["out1"], \
                        row["out2"], \
                        row["out3"],
                        row["out4"],
                        row["out5"]]))
    return lp
   
HATE=rescaledData.rdd.map(transformToLabeledPoint1)
HATEDf = SpSession.createDataFrame(HATE,["label", "features"])

INSULT=rescaledData.rdd.map(transformToLabeledPoint2)
INSULTDf = SpSession.createDataFrame(INSULT,["label", "features"])

OBSCENCE=rescaledData.rdd.map(transformToLabeledPoint3)
OBSENCEDf = SpSession.createDataFrame(OBSCENCE,["label", "features"])

SEVERE_TOXIC=rescaledData.rdd.map(transformToLabeledPoint4)
SEVERE_TOXICDf = SpSession.createDataFrame(SEVERE_TOXIC,["label", "features"])

THREAT= rescaledData.rdd.map(transformToLabeledPoint5)
THREATDf = SpSession.createDataFrame(THREAT,["label", "features"])

TOXIC=rescaledData.rdd.map(transformToLabeledPoint6)
TOXICDf = SpSession.createDataFrame(TOXIC,["label", "features"])


"""----------------------------------------------------------------------------
Perform Machine Learning
---------------------------------------------------------------------------"""
##################################################Logistic Regression################################
######################################HATE as the output
#Split into training and testing data
(trainingData, testData) = HATEDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Create the model
Classifer =  LogisticRegression(regParam=0.0,labelCol="label",\
                featuresCol="features")
Model = Classifer.fit(trainingData)

#Predict on the test data
predictions = Model.transform(testData)
predictions.select("prediction","label").show()

#Evaluate accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", \
                    labelCol="label")
evaluator.evaluate(predictions)      

#to check with the overfitting problem
predictions_train = Model.transform(trainingData)
predictions.select("prediction","label").show()


#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()

###################################### INSULT as the output
#Split into training and testing data
(trainingData, testData) = INSULTDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Create the model
Classifer =  LogisticRegression(regParam=0.0,labelCol="label",\
                featuresCol="features")
Model = Classifer.fit(trainingData)

Model.intercept
Model.coefficients

#Predict on the test data
predictions = Model.transform(testData)
predictions.select("prediction","label").show()

#Evaluate accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", \
                    labelCol="label")
evaluator.evaluate(predictions)      

#to check with the overfitting problem
predictions_train = Model.transform(trainingData)
predictions.select("prediction","label").show()

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()

###################################### OBSENCE as the output
#Split into training and testing data
(trainingData, testData) = OBSENCEDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Create the model
Classifer =  LogisticRegression(regParam=0.0,labelCol="label",\
                featuresCol="features")
Model = Classifer.fit(trainingData)

Model.intercept
Model.coefficients

#Predict on the test data
predictions = Model.transform(testData)
predictions.select("prediction","label").show()

#Evaluate accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", \
                    labelCol="label")
evaluator.evaluate(predictions)      

#to check with the overfitting problem
predictions_train = Model.transform(trainingData)
predictions.select("prediction","label").show()

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()

###################################### SEVERE_TOXIC as the output
#Split into training and testing data
(trainingData, testData) = SEVERE_TOXICDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Create the model
Classifer =  LogisticRegression(regParam=0.0,labelCol="label",\
                featuresCol="features")
Model = Classifer.fit(trainingData)

Model.intercept
Model.coefficients

#Predict on the test data
predictions = Model.transform(testData)
predictions.select("prediction","label").show()

#Evaluate accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", \
                    labelCol="label")
evaluator.evaluate(predictions)    

#to check with the overfitting problem
predictions_train = Model.transform(trainingData)
predictions.select("prediction","label").show()  

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()

###################################### THREAT as the output
#Split into training and testing data
(trainingData, testData) = THREATDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Create the model
Classifer =  LogisticRegression(regParam=0.0,labelCol="label",\
                featuresCol="features")
Model = Classifer.fit(trainingData)

Model.intercept
Model.coefficients

#Predict on the test data
predictions = Model.transform(testData)
predictions.select("prediction","label").show()

#Evaluate accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", \
                    labelCol="label")
evaluator.evaluate(predictions)      

#to check with the overfitting problem
predictions_train = Model.transform(trainingData)
predictions.select("prediction","label").show()

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()

###################################### TOXIC as the output
#Split into training and testing data
(trainingData, testData) = TOXICDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Create the model
Classifer = LogisticRegression(regParam=0.0,labelCol="label",\
                featuresCol="features")
Model = Classifer.fit(trainingData)

Model.intercept
Model.coefficients

#Predict on the test data
predictions = Model.transform(testData)
predictions.select("prediction","label").show()

#Evaluate accuracy
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", \
                    labelCol="label")
evaluator.evaluate(predictions)   

#to check with the overfitting problem
predictions_train = Model.transform(trainingData)
predictions.select("prediction","label").show()   

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()




##################################################Decision Tree################################
######################################HATE as the output
#Split into training and testing data
(trainingData, testData) = HATEDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Create the model
dtClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="label",\
                featuresCol="features")
dtModel = dtClassifer.fit(trainingData)


#Predict on the test data
predictions = dtModel.transform(testData)
predictions.select("prediction","species","label").show()

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()


###################################### INSULT as the output
#Split into training and testing data
(trainingData, testData) = INSULTDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Create the model
dtClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="label",\
                featuresCol="features")
dtModel = dtClassifer.fit(trainingData)

dtModel.numNodes
dtModel.depth

#Predict on the test data
predictions = dtModel.transform(testData)
predictions.select("prediction","species","label").show()

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()


###################################### OBSENCE as the output
#Split into training and testing data
(trainingData, testData) = OBSENCEDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Create the model
dtClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="label",\
                featuresCol="features")
dtModel = dtClassifer.fit(trainingData)

dtModel.numNodes
dtModel.depth

#Predict on the test data
predictions = dtModel.transform(testData)
predictions.select("prediction","species","label").show()

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()


###################################### SEVERE_TOXIC as the output
#Split into training and testing data
(trainingData, testData) = SEVERE_TOXICDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Create the model
dtClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="label",\
                featuresCol="features")
dtModel = dtClassifer.fit(trainingData)

dtModel.numNodes
dtModel.depth

#Predict on the test data
predictions = dtModel.transform(testData)
predictions.select("prediction","species","label").show()

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()


###################################### THREAT as the output
#Split into training and testing data
(trainingData, testData) = THREATDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Create the model
dtClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="label",\
                featuresCol="features")
dtModel = dtClassifer.fit(trainingData)

dtModel.numNodes
dtModel.depth

#Predict on the test data
predictions = dtModel.transform(testData)
predictions.select("prediction","species","label").show()

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()


###################################### TOXIC as the output
#Split into training and testing data
(trainingData, testData) = TOXICDf.randomSplit([0.75, 0.25])
trainingData.count()
testData.count()
testData.show()

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Create the model
dtClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="label",\
                featuresCol="features")
dtModel = dtClassifer.fit(trainingData)

dtModel.numNodes
dtModel.depth

#Predict on the test data
predictions = dtModel.transform(testData)
predictions.select("prediction","species","label").show()

#Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
evaluator.evaluate(predictions)      

#Draw a confusion matrix
predictions.groupBy("label","prediction").count().show()
