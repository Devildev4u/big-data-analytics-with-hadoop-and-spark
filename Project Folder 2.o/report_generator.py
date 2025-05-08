import logging
from pyspark.sql import DataFrame

def generate_report(df: DataFrame, output_path: str = "report.txt"):
    logging.info("\n========== STARTING: Report Generation ==========")
    try:
        logging.info("Converting predictions to pandas DataFrame...")
        pdf = df.toPandas()

        logging.info("Calculating statistics...")
        mean_pred = pdf['prediction'].mean()
        report = f"Prediction Summary:\nMean Prediction: {mean_pred:.2f}\n"

        logging.info(f"Writing report to {output_path}...")
        with open(output_path, "w") as f:
            f.write(report)

        logging.info(f"Report successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        raise
    logging.info("========== COMPLETED: Report Generation ==========\n")
