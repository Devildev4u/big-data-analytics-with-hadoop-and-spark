import logging
import time
from data_ingestion import load_data
from data_processing import process_data
from machine_learning import train_model
from data_analysis import analyze_data
from visualization import visualize_results
from report_generator import generate_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),  # Save to file
        logging.StreamHandler()              # Also show in console
    ]
)

def log_stage(stage):
    logging.info(f"\n========== STARTING: {stage} ==========")

def log_stage_done(stage):
    logging.info(f"\n========== COMPLETED: {stage} ==========")

def main():
    start_time = time.time()
    log_stage("Pipeline")

    log_stage("Loading and transforming data")
    df_powerlifting, _ = load_data()
    df = df_powerlifting.select("Name", "BodyweightKg", "TotalKg") \
        .withColumnRenamed("Name", "id") \
        .withColumnRenamed("BodyweightKg", "feature1") \
        .withColumnRenamed("TotalKg", "feature2") \
        .dropna()
    log_stage_done("Loading and transforming data")

    log_stage("Processing data")
    df_transformed = process_data(df)
    log_stage_done("Processing data")

    log_stage("Training model")
    predictions = train_model(df_transformed)
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

    duration = time.time() - start_time
    logging.info(f"\n========== PIPELINE FINISHED in {duration:.2f} seconds ==========")

if __name__ == "__main__":
    main()
