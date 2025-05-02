import logging
from pyspark.sql import SparkSession, DataFrame

logging.basicConfig(level=logging.INFO)

def load_data(file1: str = "openpowerlifting.csv", file2: str = "meets.csv") -> tuple[DataFrame, DataFrame]:
    try:
        logging.info("Creating Spark session...")
        spark = SparkSession.builder.appName("BigDataProject").getOrCreate()

        logging.info(f"Loading datasets: {file1}, {file2}")
        df1 = spark.read.csv(file1, header=True, inferSchema=True)
        df2 = spark.read.csv(file2, header=True, inferSchema=True)

        logging.info("Datasets loaded successfully.")
        return df1, df2

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
