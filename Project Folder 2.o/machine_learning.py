from pyspark.ml.regression import LinearRegression

def train_model(df_transformed):
    lr = LinearRegression(featuresCol="features", labelCol="feature1", regParam=0.1)
    model = lr.fit(df_transformed)
    predictions = model.transform(df_transformed)
    return predictions
