import sys
from joblib import load
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

# Load the Random Forest model
model_filename = "random_forest_model.joblib"
rf = load(model_filename)

# Initialize the Spark session
spark = SparkSession.builder.appName("RandomForestHadoop").getOrCreate()

# Read the input data from HDFS
input_data_path = sys.argv[1]
data = spark.read.csv(input_data_path, header=True, inferSchema=True)

# Convert the input data to a format suitable for the model
features = data.rdd.map(lambda row: Vectors.dense(row[:-1])).collect()

# Make predictions using the Random Forest model
predictions = rf.predict(features)

# Write the predictions to HDFS
output_data_path = sys.argv[2]
predictions_df = spark.createDataFrame(zip(data.rdd.map(lambda row: row[-1]).collect(), predictions), schema=["label", "prediction"])
predictions_df.write.csv(output_data_path, header=True)

# Stop the Spark session
spark.stop()