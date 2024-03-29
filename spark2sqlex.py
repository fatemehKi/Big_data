#Initialize SparkSession and SparkContext
from pyspark.sql import SparkSession

#Create a Spark Session
SpSession = SparkSession \
    .builder \
    .master("local") \
    .appName("py_spark") \
    .getOrCreate()

SpContext = SpSession.sparkContext

#Create a data frame from a JSON file
empDf = SpSession.read.json("customerData.json")
empDf.show()
empDf.printSchema()

#Do Data Frame queries
empDf.select("name").show()
empDf.filter(empDf["age"] == 40).show()
empDf.groupBy("gender").count().show()
empDf.groupBy("deptid").\
    agg({"salary": "avg", "age": "max"}).show()

#create a data frame from a list
deptList = [{'name': 'Sales', 'id': "100"},\
     { 'name':'Engineering','id':"200" }]
deptDf = SpSession.createDataFrame(deptList)
deptDf.show()
 
#join the data frames
empDf.join(deptDf, empDf.deptid == deptDf.id).show()
 
#cascading operations
empDf.filter(empDf["age"] >30).join(deptDf, \
        empDf.deptid == deptDf.id).\
        groupBy("deptid").\
        agg({"salary": "avg", "age": "max"}).show()
        
#............................................................................
##   Creating data frames from RDD
#............................................................................

from pyspark.sql import Row
lines = SpContext.textFile("auto-data.csv")
#remove the first line
datalines = lines.filter(lambda x: "FUELTYPE" not in x)
datalines.count()

parts = datalines.map(lambda l: l.split(","))
autoMap = parts.map(lambda p: Row(make=p[0],\
         body=p[4], hp=int(p[7])))
# Infer the schema, and register the DataFrame as a table.
autoDf = SpSession.createDataFrame(autoMap)
autoDf.show()

#............................................................................
##   Creating data frames directly from CSV
#...........................................................................
autoDf1 = SpSession.read.csv("auto-data.csv",header=True)
autoDf1.show()

#............................................................................
##   Creating and working with Temp Tables
#............................................................................

autoDf.createTempView("autos")
SpSession.sql("select * from autos where hp > 200").show()

#register a data frame as table and run SQL statements against it
empDf.createTempView("employees")
SpSession.sql("select * from employees where salary > 4000").show()
