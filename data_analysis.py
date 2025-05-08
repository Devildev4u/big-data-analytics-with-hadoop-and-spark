def analyze_data(df):
    df.select("id", "prediction").show()
