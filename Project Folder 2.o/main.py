from data_ingestion import load_data
from data_processing import process_data
from machine_learning import train_model
from data_analysis import analyze_data
from visualization import visualize_results
from report_generator import generate_report
import logging

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Pipeline started.")
    df_powerlifting, df_meets = load_data()
    

    df = df_powerlifting.select("Name", "BodyweightKg", "TotalKg") \
    .withColumnRenamed("Name", "id") \
    .withColumnRenamed("BodyweightKg", "feature1") \
    .withColumnRenamed("TotalKg", "feature2") \
    .dropna()

    df_transformed = process_data(df)
    predictions = train_model(df_transformed)
    analyze_data(predictions)
    visualize_results(predictions)
    generate_report(predictions)
    logging.info("Pipeline finished.")

if __name__ == "__main__":
    main()
