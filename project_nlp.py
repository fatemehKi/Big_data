# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:08:28 2019

@author: fkiai
"""

from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from functools import reduce
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from pyspark.sql import Row



SpSession = SparkSession.builder.master("local").appName("py_spark").getOrCreate()
SpContext = SpSession.sparkContext
datalines = SpContext.textFile("sparkml/Comment_Classification_WOH.csv")
parts=datalines.map(lambda l:l.split(','))
cmnt2=parts.map(lambda p:Row(Comment=p[0]))
cmnt3=parts.map(lambda p:Row(out1=int(p[2]), out2=int(p[3]), out3=int(p[4]), out4=int(p[6]), out5=int(p[7]), out6=int(p[8])))

parts=datalines.map(lambda l:l.split(','))
cmnt2=parts.map(lambda p:Row(Comment=p[0]))
cmnt3=parts.map(lambda p:Row(out1=int(p[2]), out2=int(p[3]), out3=int(p[4]), out4=int(p[6]), out5=int(p[7]), out6=int(p[8])))

dataset2=SpSession.createDataFrame(cmnt2)

reviews_rdd = dataset2.select("Comment").rdd.flatMap(lambda x: x)
lowerCase_sentRDD=reviews_rdd

def sent_TokenizeFunct(x):
    return nltk.sent_tokenize(x)
sentenceTokenizeRDD = lowerCase_sentRDD.map(sent_TokenizeFunct)

def word_TokenizeFunct(x):
    splitted = [word for line in x for word in line.split()]
    return splitted
wordTokenizeRDD = sentenceTokenizeRDD.map(word_TokenizeFunct)

def removeStopWordsFunct(x):
    from nltk.corpus import stopwords
    stop_words=set(stopwords.words('english'))
    filteredSentence = [w for w in x if not w in stop_words]
    return filteredSentence
stopwordRDD = wordTokenizeRDD.map(removeStopWordsFunct)

def removePunctuationsFunct(x):
    list_punct=list(string.punctuation)
    filtered = [''.join(c for c in s if c not in list_punct) for s in x]
    filtered_space = [s for s in filtered if s] #remove empty space
    return filtered_space
rmvPunctRDD = stopwordRDD.map(removePunctuationsFunct)

def lemmatizationFunct(x):
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    finalLem = [lemmatizer.lemmatize(s) for s in x]
    return finalLem
lem_wordsRDD = rmvPunctRDD.map(lemmatizationFunct)

def joinTokensFunct(x):
    joinedTokens_list = []
    x = " ".join(x)
    return x
joinedTokens = lem_wordsRDD.map(joinTokensFunct)

