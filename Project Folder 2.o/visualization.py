import matplotlib.pyplot as plt

def visualize_results(predictions):
    # Sample 100 rows for plotting
    pdf = predictions.sample(fraction=0.001).toPandas()

    # Sort by prediction for clarity
    pdf = pdf.sort_values(by="prediction", ascending=False)

    # Limit to top 50 for readability
    pdf = pdf.head(50)

    # Plot as a horizontal bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(pdf['id'], pdf['prediction'], color='skyblue')
    plt.xlabel("Prediction")
    plt.ylabel("ID")
    plt.title("Top 50 Predictions")
    plt.tight_layout()
    plt.show()
