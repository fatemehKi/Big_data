"""
Created on Mon July 24 03:38:53 2019
@author: Fatemeh Kiaie
@description: the purpose of the project is to implemented to categorize the toxic 
comments based on the types of toxicity. Examples of toxicity types can be toxic, severely toxic, obscene, threat, insult, identity hate. Two machine learning
techniques like Logistic Regression and Decision Tree are implemented to determine the 6 types of toxic
comments
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
from pyspark.ml.stat import Correlation ##################not working in version 1
from pyspark.ml.linalg import Vectors

SpSession = SparkSession.builder.master("local").appName("py_spark").getOrCreate()
SpContext = SpSession.sparkContext


datalines = SpContext.textFile("niloo_fatemeh/Comment_Classification_WOH.csv)

"""----------------------------------------------------------------------------
Cleanup Data
----------------------------------------------------------------------------"""
parts=datalines.map(lambda l:l.split(','))
cmnt2=parts.map(lambda p:Row(Comment=p[0]))
cmnt3=parts.map(lambda p:Row(hate_=int(p[2]), insult_=int(p[3]), obsence_=int(p[4]), severe_toxic_=int(p[6]), thread_=int(p[7]), toxic_=int(p[8])))

dataset2=SpSession.createDataFrame(cmnt2) ##dataset for the comments
dataset3=SpSession.createDataFrame(cmnt3) ###all output columns-multi label classification

Comments_cl = dataset2.select("Comment").rdd.flatMap(lambda x: x)

############################# Lower Case--- the text are already lower cased.. not required to implement any function
lowerCase_sentRDD=Comments_cl

############################# Tokenization
def sent_TokenizeFunct(x):
    return nltk.sent_tokenize(x)
sentenceTokenizeRDD = lowerCase_sentRDD.map(sent_TokenizeFunct)
sentenceTokenizeRDD.take(5)

def word_TokenizeFunct(x):
    splitted = [word for line in x for word in line.split()]
    return splitted
wordTokenizeRDD = sentenceTokenizeRDD.map(word_TokenizeFunct)
wordTokenizeRDD.take(5)

############################# Removing Stop Words
def removeStopWordsFunct(x):
    #from nltk.corpus import stopwords
    stop_words=set(stopwords.words('english'))
    filteredSentence = [w for w in x if not w in stop_words]
    return filteredSentence
stopwordRDD = wordTokenizeRDD.map(removeStopWordsFunct)
stopwordRDD.take(5)

############################# Removing Punctuations
def removePunctuationsFunct(x):
    list_punct=list(string.punctuation)
    filtered = [''.join(c for c in s if c not in list_punct) for s in x]
    filtered_space = [s for s in filtered if s] #remove empty space
    return filtered_space
rmvPunctRDD = stopwordRDD.map(removePunctuationsFunct)
rmvPunctRDD.take(5)

############################# Leminization
def lemmatizationFunct(x):
    #nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    finalLem = [lemmatizer.lemmatize(s) for s in x]
    return finalLem
lem_wordsRDD = rmvPunctRDD.map(lemmatizationFunct)
lem_wordsRDD.take(5)

############################ joining tokens.. not required.
def joinTokensFunct(x):
    #joinedTokens_list = []
    x = " ".join(x)
    return x
joinedTokens = lem_wordsRDD.map(joinTokensFunct)
df_joinedTokens = joinedTokens.map(lambda x: (x, )).toDF()
df_joinedTokens.show()

###name assignment
df_joinedTokens2 = df_joinedTokens.selectExpr("_1 as Comment")
df_joinedTokens2.printSchema()


############################ TF-IDF
df_s = lem_wordsRDD.map(lambda x: (x, )).toDF()
hashingTF = HashingTF(inputCol="_1", outputCol="rawFeatures", numFeatures=1000)
featurizedData = hashingTF.transform(df_s)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledDf = idfModel.transform(featurizedData)
df4=rescaledDf.selectExpr("features")

rdf4=df4.select("features").rdd.flatMap(lambda x: x)


############################ joining tokens.. not required.
def joinTokensFunct2(x):
    #joinedTokens_list = []
    x = " ".join(x)
    return x
joinedTokens = rdf4.map(joinTokensFunct2)
df_joined = joinedTokens.map(lambda x: (x, )).toDF()

########################### Merging rescaled data (after tf-idf) with the target dataset
df1 = dataset3.withColumn("id", monotonically_increasing_id())
df2 = rescaledDf.withColumn("id", monotonically_increasing_id())

rescaledData=df2.join(df1, "id", "outer").drop("id") 

"""--------------------------------------------------------------------------
Perform Data Analytics
-------------------------------------------------------------------------"""
############################ Correlation analysis
for i in dataset3.columns:
    if not( isinstance(dataset3.select(i).take(1)[0][0], unicode)):
        print( "Correlation to toxic for ", i, dataset3.stat.corr('toxic',i))



############################ Transform to a Data Frame for input to Machine Learing
def transformToLabeledPoint1(row):
    lp = ( row["hate_"], \
                Vectors.dense([row["features"],\
                        row["insult_"], \
                        row["obsence_"], \
                        row["severe_toxic_"],
                        row["thread_"],
                        row["toxic_"]]))
    return lp

def transformToLabeledPoint2(row):
    lp = ( row["insult_"], \
                Vectors.dense([row["features"],\
                        row["hate_"], \
                        row["obsence_"], \
                        row["severe_toxic_"],
                        row["thread_"],
                        row["toxic_"]]))
    return lp

def transformToLabeledPoint3(row):
    lp = ( row["obsence_"], \
                Vectors.dense([row["features"],\
                        row["hate_"], \
                        row["insult_"], \
                        row["severe_toxic_"],
                        row["thread_"],
                        row["toxic_"]]))
    return lp



def transformToLabeledPoint4(row):
    lp = ( row["severe_toxic_"], \
                Vectors.dense([row["features"],\
                        row["hate_"], \
                        row["insult_"], \
                        row["obsence_"],
                        row["thread_"],
                        row["toxic_"]]))
    return lp

def transformToLabeledPoint5(row):
    lp = ( row["thread_"], \
                Vectors.dense([row["features"],\
                        row["hate_"], \
                        row["insult_"], \
                        row["obsence"],
                        row["severe_toxic_"],
                        row["toxic_"]]))
    return lp

def transformToLabeledPoint6(row):
    lp = ( row["toxic_"], \
                Vectors.dense([row["features"],\
                        row["hate_"], \
                        row["insult_"], \
                        row["obsence_"],
                        row["severe_toxic_"],
                        row["thread"]]))
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
