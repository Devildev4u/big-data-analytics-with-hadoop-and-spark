import logging
from pyspark.sql import DataFrame

def generate_report(df: DataFrame, output_path: str = "report.txt"):
    try:
        logging.info("Generating model performance report...")
        pdf = df.toPandas()
        mean_pred = pdf['prediction'].mean()
        report = f"Prediction Summary:\nMean Prediction: {mean_pred:.2f}\n"

        with open(output_path, "w") as f:
            f.write(report)

        logging.info(f"Report saved to {output_path}")
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        raise
