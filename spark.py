# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:21:44 2019

@author: mfatemeh
"""

def sum(func,list):
    total=0
    for i in list:
        i=func(i)        
        total += i
    return total

sum(lambda x: x,range(1,10))
sum(lambda x:x*x, range(1,11))

m=map(lambda x: x**2, list(range(1,10))) #map will apply function on each individual items and will return as many as they were..
# it is basically transformation.. and another map function is 
#filter() in python
#The filter() method filters the given sequence with the help of a function that tests each element in the sequence to be true or not.
#syntax:
#filter(function, sequence)
#Parameters:
#function: function that tests if each element of a 
#sequence true or not.
#sequence: sequence which needs to be filtered, it can 
#be sets, lists, tuples, or containers of any iterators.
#Retruns:
#returns an iterator that is already filtered. 



###reduced functions will retun less items as it was at the beginning
reduce(lambda x, y:x+y, m)

f=filter(lambda x:x%2 == 0,m) 

l=range(1,5001)
rdd_l=sc.parallelize(l) # it is basicall turns the list l to RDD
help(sc)

rdd2=sc.textFile('hdfs:///user/fatemeh/Social_Network_Ads.csv') # turning the csv file to rdd
rdd2.take(1)
# output [u'User ID,Gender,Age,EstimatedSalary,Purchased']


KeyboardInterrupt
rdd2.take(5)
#[u'User ID,Gender,Age,EstimatedSalary,Purchased', u'15624510,Male,19,19000,0', u'15810944,Male,35,20000,0', u'15668575,Female,26,43000,0', u'15603246,Female,27,57000,0']
rdd2.take(20)
#[u'User ID,Gender,Age,EstimatedSalary,Purchased', u'15624510,Male,19,19000,0', u'15810944,Male,35,20000,0', u'15668575,Female,26,43000,0', u'15603246,Female,27,57000,0', u'15804002,Male,19,76000,0', u'15728773,Male,27,58000,0', u'15598044,Female,27,84000,0', u'15694829,Female,32,150000,1', u'15600575,Male,25,33000,0', u'15727311,Female,35,65000,0', u'15570769,Female,26,80000,0', u'15606274,Female,26,52000,0', u'15746139,Male,20,86000,0', u'15704987,Male,32,18000,0', u'15628972,Male,18,82000,0', u'15697686,Male,29,80000,0', u'15733883,Male,47,25000,1', u'15617482,Male,45,26000,1', u'15704583,Male,46,28000,1']
rdd_txt=sc.textFile('wordcount.txt')
rdd_txt.take(2)
#[u'This is a simple text file to show', u'word count implementation']
rdd_txt.collect()
#[u'This is a simple text file to show', u'word count implementation', u'in pyspark', u'pyspark is a great system', u'simple test show show', u'word more word']
rdd_words=rdd_txt.flatMap(lambda x:x.split()) #delimtor is space here
rdd_words.collect()
#[u'This', u'is', u'a', u'simple', u'text', u'file', u'to', u'show', u'word', u'count', u'implementation', u'in', u'pyspark', u'pyspark', u'is', u'a', u'great', u'system', u'simple', u'test', u'show', u'show', u'word', u'more', u'word']
rdd_words.countByValue()
#defaultdict(<type 'int'>, {u'a': 2, u'count': 1, u'great': 1, u'word': 3, u'show': 3, u'This': 1, u'text': 1, u'is': 2, u'pyspark': 2, u'system': 1, u'to': 1, u'file': 1, u'in': 1, u'test': 1, u'implementation': 1, u'simple': 2, u'more': 1})
word_count= rdd_words.countByValue() #became the dictionary

                                  Sort it?
######second method:
rdd_lines = sc.textFile('wordcount.txt')
rdd_txt.flatMap


###########
rdd_records=rdd_2.map(lambda x:x.split(','))
