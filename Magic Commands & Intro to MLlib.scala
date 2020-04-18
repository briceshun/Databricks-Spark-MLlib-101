// Databricks notebook source
// MAGIC %md <img src="https://github.com/briceshun/Databricks-Spark-MLlib-101/blob/master/Training%20-%20Magic%20Commands%20&%20Intro%20to%20Machine%20Learning.PNG?raw=true" width="1900">

// COMMAND ----------

// MAGIC %md 
// MAGIC ##<img src="https://i.vimeocdn.com/portrait/18609368_640x640" width="55"/> Magic Commands
// MAGIC 
// MAGIC Databricks is a great tool that allows for flexible coding - It allows you to code in multiple languages in a single notebook!
// MAGIC 
// MAGIC The default language for each cell is shown in ( ) next to the notebook name: <br>
// MAGIC <img src="https://github.com/briceshun/TVNZ-Training-Banners/blob/master/toolbar.png?raw=true" width="1000"/>
// MAGIC 
// MAGIC <br>
// MAGIC You can code in another supported language using the language magic command `%language`. <br>
// MAGIC Just specifying the language magic commands at the beginning of a cell.
// MAGIC 
// MAGIC Examples of most commonly used language magic commands are: 
// MAGIC * `%python`
// MAGIC * `%r`
// MAGIC * `%scala`
// MAGIC * `%sql`

// COMMAND ----------

// No Magic Command needed since this is a Scala notebook!
println("Hello Scala!")

// COMMAND ----------

// MAGIC %python
// MAGIC print("Hello Python!")

// COMMAND ----------

// MAGIC %r
// MAGIC print("Hello R!")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT 'Hello SQL!'

// COMMAND ----------

// MAGIC %md 
// MAGIC ##<img src="https://i.vimeocdn.com/portrait/18609368_640x640" width="55"/> Working with Packages
// MAGIC Most commonly used packages from `Python`, `R`, and `Scala` are already pre-installed in Databricks. You simply have to load them. <br>
// MAGIC Other less common packages can be installed on the cluster if need be.
// MAGIC 
// MAGIC <img src="https://cdn4.iconfinder.com/data/icons/interface-feedback-solid-style/24/interface-question-mark-512.png" width="25"/> You can read more about packages <a href="https://docs.databricks.com/libraries.html"/>here</a>.

// COMMAND ----------

// MAGIC %md 
// MAGIC ###<img src="https://www.stat.auckland.ac.nz/~paul/Reports/Rlogo/Rlogo.svg" width="35"/> R
// MAGIC Packages can be temporarily installed using the `install.packages()` function in-line. <br>
// MAGIC You should install them to the cluster if you want them to be permanently available. <br>

// COMMAND ----------

// MAGIC %r
// MAGIC library('SparkR')

// COMMAND ----------

// MAGIC %md 
// MAGIC ###<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1024px-Python-logo-notext.svg.png" width="30"/> Python

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql.types import *
// MAGIC from pyspark.sql.functions import *
// MAGIC from pyspark.ml import Pipeline
// MAGIC from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
// MAGIC from pyspark.ml.regression import GeneralizedLinearRegression

// COMMAND ----------

// MAGIC %md 
// MAGIC ###<img src="https://sudheerhadoopdev.files.wordpress.com/2016/07/scala-logo-256.png?w=256"  width="30"/>  Scala

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.regression._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors  
import org.apache.spark.ml.Pipeline  

// COMMAND ----------

// MAGIC %md 
// MAGIC ##<img src="https://i.vimeocdn.com/portrait/18609368_640x640" width="55"/> Machine Learning
// MAGIC <img src="https://miro.medium.com/max/3056/1*NAAPVShFg6G0oOJKvEfGLQ.jpeg" alt="ML Process" width="500">
// MAGIC <img src="https://d1jnx9ba8s6j9r.cloudfront.net/blog/wp-content/uploads/2018/03/How-Machine-Learning-Works-What-is-Machine-Learning-Edureka-1.gif" width="750">
// MAGIC 
// MAGIC **1. Data Collection**: Gathering the raw data needed for the ML process. <br>
// MAGIC **2. Data Munging**: "Massaging" the data into shape using a series of transforms. <br>
// MAGIC **3. Model Training**: The actual "learning" step. Involves feeding the model with data and tuning it. <br>
// MAGIC **4. Model Evaluation**: Assessing the model's ability to make predictions. <br>
// MAGIC **5. Model Improvement**: An iterative process involving re-engineering the data and tuning the model's parameters to maximise its predictive power. <br>
// MAGIC 
// MAGIC 
// MAGIC #### Types of Problems
// MAGIC <img src="https://miro.medium.com/max/2796/1*FUZS9K4JPqzfXDcC83BQTw.png" width="1000"/>
// MAGIC 
// MAGIC <img src="https://cdn4.iconfinder.com/data/icons/interface-feedback-solid-style/24/interface-question-mark-512.png" width="25"/> 
// MAGIC You can read more about other ML algorithms <a href="https://spark.apache.org/docs/latest/ml-classification-regression.html"/>here</a>, and <a href="https://www.edureka.co/blog/what-is-machine-learning/"/>here</a>.

// COMMAND ----------

// MAGIC %md
// MAGIC ###<img src="https://pngimage.net/wp-content/uploads/2018/05/database-logo-png-3.png" width="40"/> The Titanic Dataset
// MAGIC The *Titanic* dataset is a classic dataset used for machine learning. <br>
// MAGIC It contains a good mix of *textual*, *boolean*, *continuous*, and *categorical* variables. <br>
// MAGIC In addition, it exhibits interesting characteristics such as *missing values* and *outliers* that will allow us to demonstrate data transformations.
// MAGIC 
// MAGIC <img src="https://media1.giphy.com/media/4d8uCMJE5b8ru/source.gif" width="700"/>
// MAGIC 
// MAGIC We will create and train a **Classification** algorithm (<a href = "https://christophm.github.io/interpretable-ml-book/logistic.html">Logistic Regression</a>), and evaluate its predictive power. <br>
// MAGIC Our variable of interest in this example is `Survival`. 
// MAGIC 
// MAGIC Here’s a brief summary of the **14** variables:
// MAGIC 
// MAGIC * **`Survival`**: A boolean indicating whether the passenger survived or not (0 = No; 1 = Yes)
// MAGIC * `Pclass`: Passenger class (1 = 1st; 2 = 2nd; 3 = 3rd)
// MAGIC * `Name`: Title and family names
// MAGIC * `Sex`: Male/female
// MAGIC * `Age`: Age
// MAGIC * `Sibsp`: Number of siblings/spouses aboard
// MAGIC * `Parch`: Number of parents/children aboard
// MAGIC * `Ticket`: Ticket number.
// MAGIC * `Fare`: Passenger fare (British Pound).
// MAGIC * `Cabin`: Cabin location
// MAGIC * `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
// MAGIC 
// MAGIC 
// MAGIC <img src="https://outdoortycoon.com/wp-content/uploads/2017/01/Download-Icon-Wenzel.png" width=30 /> Click <a href="https://raw.githubusercontent.com/briceshun/Databricks-Spark-MLlib-101/master/Titanic%20Custom.csv">here</a> to download the dataset.

// COMMAND ----------

// MAGIC %md
// MAGIC #### <img src="https://pngimage.net/wp-content/uploads/2018/05/database-logo-png-3.png" width="30"/> Loading the Dataset
// MAGIC Loading your own data in Databricks can be completed in 5 steps:
// MAGIC 1. Open the *Data Tab* on the ribbon (left side of your screen)
// MAGIC 2. Click on *Add Data*
// MAGIC 3. Drag and Drop your File
// MAGIC 4. Click *Create Table in Notebook*
// MAGIC 5. Run the code

// COMMAND ----------

// MAGIC %python
// MAGIC # LOADING THE DATASET
// MAGIC # File location and type
// MAGIC file_location = "/FileStore/tables/Titanic_Custom-45472.csv"
// MAGIC file_type = "csv"
// MAGIC 
// MAGIC # CSV options
// MAGIC infer_schema = "true"
// MAGIC first_row_is_header = "true"
// MAGIC delimiter = ","
// MAGIC 
// MAGIC # The applied options are for CSV files. For other file types, these will be ignored.
// MAGIC df = spark.read.format(file_type) \
// MAGIC                .option("inferSchema", infer_schema) \
// MAGIC                .option("header", first_row_is_header) \
// MAGIC                .option("sep", delimiter) \
// MAGIC                .load(file_location)
// MAGIC 
// MAGIC display(df)
// MAGIC 
// MAGIC temp_table_name = "Titanic_Dataset"
// MAGIC df.createOrReplaceTempView(temp_table_name)

// COMMAND ----------

// MAGIC %md 
// MAGIC ##### <img src="https://cdn3.iconfinder.com/data/icons/flat-pro-basic-set-2-1/32/flag-blue-512.png" width="35"/> Remember to save your dataset as a *Temporary View* before switching between languages.
// MAGIC Creating/Replacing Temp View
// MAGIC * <img src="https://www.stat.auckland.ac.nz/~paul/Reports/Rlogo/Rlogo.svg" width="25"/> `createOrReplaceTempView(x, viewName)`
// MAGIC * <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1024px-Python-logo-notext.svg.png" width="25"/> `df.createOrReplaceTempView(viewName)`
// MAGIC * <img src="https://sudheerhadoopdev.files.wordpress.com/2016/07/scala-logo-256.png?w=256"  width="25"/> `df.createOrReplaceTempView(viewName)`
// MAGIC 
// MAGIC Reading Temp View using SQL
// MAGIC * <img src="https://www.stat.auckland.ac.nz/~paul/Reports/Rlogo/Rlogo.svg" width="25"/> `sql(SelectStatement)`
// MAGIC * <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1024px-Python-logo-notext.svg.png" width="25"/> `spark.sql(SelectStatement)`
// MAGIC * <img src="https://sudheerhadoopdev.files.wordpress.com/2016/07/scala-logo-256.png?w=256"  width="25"/> `spark.sql(SelectStatement)`

// COMMAND ----------

// MAGIC %md 
// MAGIC ###<img src="https://www.stat.auckland.ac.nz/~paul/Reports/Rlogo/Rlogo.svg" width="35"/>  SparkR

// COMMAND ----------

// MAGIC %r
// MAGIC # 1 DATA COLLECTION
// MAGIC # 1.1 Read Table
// MAGIC df <- sql("select * from Titanic_Dataset")
// MAGIC display(df)
// MAGIC 
// MAGIC # 1.2 Train-Test Split
// MAGIC train <- filter(df, df$PassengerId <= 600)
// MAGIC test <- filter(df, df$PassengerId > 600)
// MAGIC #labels <- select(test, c("PassengerId", "Survived"))
// MAGIC #test <- drop(test, "Survived")
// MAGIC 
// MAGIC createOrReplaceTempView(df, "TVNZ")

// COMMAND ----------

// MAGIC %r
// MAGIC # 2 DATA MUNGING
// MAGIC # 2.1 Calculate Measures of Spread
// MAGIC mu_Age <- head(select(train, mean(train$Age)))[[1]]
// MAGIC sd_Age <- head(select(train, sd(train$Age)))[[1]]
// MAGIC LB_Age <- mu_Age - (2*sd_Age)
// MAGIC UB_Age <- mu_Age + (2*sd_Age)
// MAGIC                
// MAGIC mu_Fare <- head(select(train, mean(train$Fare)))[[1]]
// MAGIC sd_Fare <- head(select(train, sd(train$Fare)))[[1]]
// MAGIC LB_Fare <- mu_Fare - (2*sd_Fare)
// MAGIC UB_Fare <- mu_Fare + (2*sd_Fare)
// MAGIC 
// MAGIC 
// MAGIC DataMunging <- function(data)
// MAGIC               {dataset <- data
// MAGIC               
// MAGIC                # 2.2 Replacing Missing Values with Mean
// MAGIC                dataset <- fillna(dataset, mu_Age, "Age")
// MAGIC                dataset <- fillna(dataset, mu_Fare, "Fare")
// MAGIC                
// MAGIC                # 2.3 Identifying and Replacing Outliers with LB and UB
// MAGIC                dataset <- withColumn(dataset, "Age", ifelse(dataset$Age > UB_Age, UB_Age, dataset$Age))
// MAGIC                dataset <- withColumn(dataset, "Age", ifelse(dataset$Age < LB_Age, LB_Age, dataset$Age))
// MAGIC                dataset <- withColumn(dataset, "Fare", ifelse(dataset$Age > UB_Fare, UB_Fare, dataset$Age))
// MAGIC                dataset <- withColumn(dataset, "Fare", ifelse(dataset$Age < LB_Fare, LB_Fare, dataset$Age))
// MAGIC               }
// MAGIC 
// MAGIC # 2.4 Apply Transformations to datasets
// MAGIC train2 <- DataMunging(train)
// MAGIC test2 <- DataMunging(test)

// COMMAND ----------

// MAGIC %r
// MAGIC # 3. MODEL TRAINING
// MAGIC model_train <- spark.glm(train2, Survived ~ Sex + Age + SibSp + Parch + Fare, family=binomial)
// MAGIC model_summary <- summary(model_train)
// MAGIC model_summary

// COMMAND ----------

// MAGIC %r
// MAGIC # 4. MODEL EVALUATION
// MAGIC # 4.1 Predict
// MAGIC predictions <- predict(model_train, test2)
// MAGIC 
// MAGIC # 4.2 Evaluate
// MAGIC #predictions <- join(predictions, labels, predictions$PassengerId == labels$PassengerId, "left")
// MAGIC predictions <- withColumn(predictions, "Predicted", ifelse(predictions$prediction > 0.5, 1, 0))
// MAGIC predictions <- withColumn(predictions, "Accuracy", ifelse(predictions$predicted == predictions$Survived, 1, 0))
// MAGIC 
// MAGIC Accuracy <- head(select(predictions, sum(predictions$Accuracy)))[[1]]/nrow(predictions)
// MAGIC cat("The R GLR Model's accuracy is", round(Accuracy, 4)*100, "%")

// COMMAND ----------

// MAGIC %r
// MAGIC display(arrange(select(predictions, c("PassengerId", "prediction", "Predicted", "Survived", "Accuracy")), "prediction"))

// COMMAND ----------

// MAGIC %md 
// MAGIC ###<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1024px-Python-logo-notext.svg.png" width="30"/> PySpark

// COMMAND ----------

// MAGIC %python
// MAGIC # 1 DATA COLLECTION
// MAGIC # 1.1 Read Table
// MAGIC #display(df)
// MAGIC df = spark.sql("SELECT * FROM TVNZ")
// MAGIC 
// MAGIC # 1.2 Train-Test Split
// MAGIC train = df.filter(df.PassengerId <= 600)
// MAGIC test = df.filter(df.PassengerId > 600)
// MAGIC 
// MAGIC display(df)

// COMMAND ----------

// MAGIC %python
// MAGIC # 2 DATA MUNGING
// MAGIC # 2.1 Select Columns
// MAGIC columns = ["PassengerId", "Age", "Sex", "SibSp", "Parch", "Fare", "Survived"]
// MAGIC train2 = train.select(columns)
// MAGIC test2 = test.select(columns)
// MAGIC 
// MAGIC def DataMunging(dataset, column):
// MAGIC   # 2.2 Calculate Measures of Spread
// MAGIC   train_stats = train2.select(mean(col(column)).alias('mu'),
// MAGIC                               stddev(col(column)).alias('sd')
// MAGIC                              ).collect()
// MAGIC 
// MAGIC   mu = train_stats[0]['mu']
// MAGIC   sd = train_stats[0]['sd']
// MAGIC   LB = mu - (2*sd)
// MAGIC   UB = mu + (2*sd)
// MAGIC   
// MAGIC   # 2.2 Replacing Missing Values with Mean
// MAGIC   dataset = dataset.fillna({column:mu})
// MAGIC                              
// MAGIC   # 2.3 Identifying and Replacing Outliers with LB and UB
// MAGIC   dataset = dataset.withColumn(column,  when(dataset[column] < LB, LB).when(dataset[column] > UB, UB).otherwise(dataset[column]))
// MAGIC   
// MAGIC   return dataset
// MAGIC  
// MAGIC # 2.4 Apply Transformations to datasets
// MAGIC train2 = DataMunging(DataMunging(train2, "Age"), "Fare")
// MAGIC test2 = DataMunging(DataMunging(test2, "Age"), "Fare")
// MAGIC 
// MAGIC 
// MAGIC # 2.5 Convert to Vector
// MAGIC def get_dummy(df,indexCol,categoricalCols,continuousCols,labelCol):
// MAGIC     indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
// MAGIC                  for c in categoricalCols ]
// MAGIC 
// MAGIC     # default setting: dropLast=True
// MAGIC     encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
// MAGIC                  outputCol="{0}_encoded".format(indexer.getOutputCol()))
// MAGIC                  for indexer in indexers ]
// MAGIC 
// MAGIC     assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
// MAGIC                                 + continuousCols, outputCol="features")
// MAGIC 
// MAGIC     pipeline = Pipeline(stages=indexers + encoders + [assembler])
// MAGIC 
// MAGIC     model=pipeline.fit(df)
// MAGIC     data = model.transform(df)
// MAGIC 
// MAGIC     if labelCol == "Survived":
// MAGIC         data = data.withColumn('label',col(labelCol))
// MAGIC     else:
// MAGIC         data = data.withColumn('label',lit(1))
// MAGIC 
// MAGIC         
// MAGIC     if indexCol:
// MAGIC         return data.select(indexCol,'features','label')
// MAGIC     else:
// MAGIC         return data.select('features','label')
// MAGIC 
// MAGIC train3 = get_dummy(train2,"PassengerId", ["Sex"], ["Age", "SibSp", "Parch", "Fare"], "Survived")
// MAGIC test3 = get_dummy(test2,"PassengerId", ["Sex"], ["Age", "SibSp", "Parch", "Fare"], "Survived")

// COMMAND ----------

// MAGIC %python
// MAGIC display(train3)

// COMMAND ----------

// MAGIC %md 
// MAGIC ##### <img src="https://cdn3.iconfinder.com/data/icons/flat-pro-basic-set-2-1/32/flag-blue-512.png" width="35"/> Vectorisation
// MAGIC The data munging step in `python` is slightly less forward than in `R`. <br>
// MAGIC The dataframe has to be converted into a vector before being passed into the algorithm. <br>
// MAGIC *See Step 2.5 Convert to Vector*

// COMMAND ----------

// MAGIC %python
// MAGIC # 3. MODEL TRAINING
// MAGIC # 3.1 Initialise Model
// MAGIC glr = GeneralizedLinearRegression(family="Binomial")
// MAGIC 
// MAGIC # 3.2 Fit the model
// MAGIC model = glr.fit(train3)
// MAGIC model.summary

// COMMAND ----------

// MAGIC %python
// MAGIC # 4. MODEL EVALUATION
// MAGIC # 4.1 Predict
// MAGIC predictions = model.transform(test3)
// MAGIC 
// MAGIC # 4.2 Evaluate
// MAGIC predictions = predictions.withColumn("Predicted", when(predictions["prediction"] > 0.5, 1).otherwise(0))
// MAGIC predictions = predictions.withColumn("Correct", when(predictions["predicted"] == predictions["label"], 1).otherwise(0))
// MAGIC 
// MAGIC Accuracy = predictions.groupBy().sum().collect()[0]["sum(Correct)"]/predictions.groupBy().count().collect()[0]["count"]
// MAGIC 
// MAGIC print("The Python GLR Model's accuracy is {:0.2f}%".format(Accuracy*100))

// COMMAND ----------

// MAGIC %python
// MAGIC display(predictions.select("PassengerID", "prediction", "Predicted", "Label"))

// COMMAND ----------

// MAGIC %md 
// MAGIC ###<img src="https://sudheerhadoopdev.files.wordpress.com/2016/07/scala-logo-256.png?w=256"  width="40"/>  Scala

// COMMAND ----------

// 1 DATA COLLECTION
// 1.1 Read Table
val df = spark.sql("select * from Titanic_Dataset")
display(df)

// 1.2 Train-Test Split
var train = df.filter($"PassengerId" <= 600)
var test = df.filter($"PassengerId" > 600)

// COMMAND ----------

// 2 DATA MUNGING
// 2.1 Select Columns
var cols = List("PassengerId", "Age", "Sex", "SibSp", "Parch", "Fare", "Survived")
var train2 = train.select(cols.map(train(_)): _*)
var test2 = test.select(cols.map(test(_)): _*)

def DataMunging(df: DataFrame, column: String): DataFrame = {
  var dataset = df 
  var c = column
  
  // 2.2 Calculate Measures of Spread
  var train_stats = train2.select(avg(column) as "mu",
                                  stddev(column) as "sd")
  var mu = train_stats.head().getDouble(0)
  var sd = train_stats.head().getDouble(1)
  var LB = mu - (2*sd)
  var UB = mu + (2*sd)
  
  // 2.3 Replacing Missing Values with Mean & Identifying and Replacing Outliers with LB and UB
  var dataset1 = dataset.withColumn(column, when(dataset.col(column).isNull, s"""$mu""").when(dataset.col(column) < s"""$LB""", s"""$LB""").when(dataset.col(column) > s"""$UB""", s"""$UB""").otherwise(dataset.col(column)))
  
  var dataset2 = dataset1.withColumn(column, dataset1(column).cast("Double"))
  
  dataset2
}
 
// 2.4 Apply Transformations to datasets
var train3 = DataMunging(DataMunging(train2, "Age"), "Fare")
var test3 = DataMunging(DataMunging(test2, "Age"), "Fare")

// 2.5 Convert to Vector
def get_dummy(df: DataFrame, indexCol: String, categoricalCols: Array[String], continuousCols: Array[String], labelCol: String): DataFrame = {
  val indexers = categoricalCols.map( c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_indexed") )
  val encoders = categoricalCols.map( c => new OneHotEncoder().setInputCol(s"${c}_indexed").setOutputCol(s"${c}_encoded") )
  val pipeline = new Pipeline().setStages(indexers ++ encoders)
  val transformed = pipeline.fit(df).transform(df)
  
  val vec_features = transformed.columns.filter(_.contains("_encoded")).toArray
  val assembler = new VectorAssembler().setInputCols(vec_features ++ continuousCols).setOutputCol("features")
  val pipelineassember = new Pipeline().setStages(Array(assembler))
  
  val dataset = pipelineassember.fit(transformed).transform(transformed)
  
  
  val data = if(labelCol == "Survived"){dataset.withColumn("label",dataset("Survived"))} 
             else{dataset.withColumn("label",lit(1))}
  
  val final_data = if(indexCol == "PassengerId"){data.select($"PassengerId", $"features", $"label")} 
                   else{data.select($"features", $"label")}
  
  final_data
}

var train4 = get_dummy(train3, "PassengerId", Array("Sex"), Array("Age", "SibSp", "Parch", "Fare"), "Survived")
var test4 = get_dummy(test3, "PassengerId", Array("Sex"), Array("Age", "SibSp", "Parch", "Fare"), "Survived")

// COMMAND ----------

// MAGIC %md 
// MAGIC ##### <img src="https://cdn3.iconfinder.com/data/icons/flat-pro-basic-set-2-1/32/flag-blue-512.png" width="35"/> Vectorisation
// MAGIC Like in `python`, the `scala` dataframe has to be converted into a vector before being passed into the algorithm. <br>
// MAGIC *See Step 2.5 Convert to Vector*

// COMMAND ----------

// 3. MODEL TRAINING
// 3.1 Initialise Model
val glr = new GeneralizedLinearRegression().setFamily("binomial")
// 3.2 Fit the model
val model = glr.fit(train4)
model.summary

// COMMAND ----------

// 4. MODEL EVALUATION
// 4.1 Predict
var predictions = model.transform(test4)

// 4.2 Evaluate
var predictions1 = predictions.withColumn("Predicted", when($"prediction" > 0.5, 1).otherwise(0))
var predictions2 = predictions1.withColumn("Correct", when($"predicted" === $"label", 1).otherwise(0))

//Accuracy = predictions.groupBy().sum().collect()[0]["sum(Correct)"]/predictions.groupBy().count().collect()[0]["count"]

val Accuracy = BigDecimal((predictions2.agg(sum("Correct"),count("PassengerId"))
                                       .withColumn("Accuracy", format_number($"sum(Correct)"/$"count(PassengerId)", 4).cast("Double"))
                                       .head().getDouble(2)) * 100
                         ).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble

// COMMAND ----------

println(s"The Scala GLR Model's accuracy is ${Accuracy} %")

// COMMAND ----------

display(predictions2.select($"PassengerID", $"prediction", $"Predicted", $"Label"))

// COMMAND ----------

// MAGIC %md 
// MAGIC ##<img src="https://i.vimeocdn.com/portrait/18609368_640x640" width="55"/> Summary
// MAGIC 
// MAGIC **1. Magic Commands**
// MAGIC   * `%language`: switch between languages
// MAGIC   * `%md`: switch to markdown
// MAGIC   * `createOrReplaceTempView`: create a temporary view before switching between languages
// MAGIC 
// MAGIC **2. Packages**
// MAGIC   * Popular packages pre-installed
// MAGIC   * Packages can be permanently installed on cluster
// MAGIC   * R packages installed using `install.packages()` in-line are temporary
// MAGIC   
// MAGIC **3. Machine Learning**
// MAGIC   * ***Steps:***
// MAGIC     1. Data Collection
// MAGIC     2. Data Munging
// MAGIC     3. Model Training
// MAGIC     4. Model Evaluation
// MAGIC     5. Model Improvement
// MAGIC   * ***Problem Types:***
// MAGIC     1. Supervised: Regression, Classification
// MAGIC     2. Unsupervised: Clustering
// MAGIC   * ***Learning Curves:***
// MAGIC     * Data Preprocessing: `R`
// MAGIC     * Vectorisation: `Scala` and `Python`
// MAGIC   
