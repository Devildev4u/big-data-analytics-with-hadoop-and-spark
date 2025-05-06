import matplotlib.pyplot as plt

def visualize_results(predictions):
    pdf = predictions.toPandas()
    plt.scatter(pdf['id'], pdf['prediction'], label='Predictions')
    plt.legend()
    plt.xlabel("ID")
    plt.ylabel("Prediction")
    plt.show()
