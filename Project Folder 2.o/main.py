import os
import warnings
import logging
from data_ingestion import load_data
from data_processing import process_data
from machine_learning import train_model
from data_analysis import analyze_data
from visualization import visualize_results
from report_generator import generate_report

# ============================== #
#        GLOBAL CONFIG SETUP     #
# ============================== #

# Suppress unnecessary warnings
os.environ["PYSPARK_VERBOSE"] = "false"
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"
os.environ["HADOOP_HOME"] = "C:/hadoop"  # Optional, if using Hadoop

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("org").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ============================== #
#        LOGGING HELPERS        #
# ============================== #

def log_stage(stage_name):
    logging.info(f"\n========== STARTING: {stage_name} ==========")

def log_stage_done(stage_name):
    logging.info(f"\n========== COMPLETED: {stage_name} ==========")

# ============================== #
#         MAIN PIPELINE         #
# ============================== #

def main():
    log_stage("Pipeline")

    log_stage("Loading data")
    df_powerlifting, df_meets = load_data()
    logging.info(f"Data loaded: df_powerlifting with {df_powerlifting.count()} rows, df_meets with {df_meets.count()} rows.")
    log_stage_done("Loading data")

    log_stage("Transforming data")
    df = df_powerlifting.select("Name", "BodyweightKg", "TotalKg") \
        .withColumnRenamed("Name", "id") \
        .withColumnRenamed("BodyweightKg", "feature1") \
        .withColumnRenamed("TotalKg", "feature2") \
        .dropna()
    df.show(5)
    log_stage_done("Transforming data")

    log_stage("Processing data")
    df_transformed = process_data(df)
    df_transformed.show(5)
    log_stage_done("Processing data")

    log_stage("Training model")
    predictions = train_model(df_transformed)
    predictions.show(5)
    log_stage_done("Training model")

    log_stage("Analyzing data")
    analyze_data(predictions)
    log_stage_done("Analyzing data")

    log_stage("Visualizing results")
    visualize_results(predictions)
    log_stage_done("Visualizing results")

    log_stage("Generating report")
    generate_report(predictions)
    log_stage_done("Generating report")

    log_stage_done("Pipeline")

if __name__ == "__main__":
    main()
