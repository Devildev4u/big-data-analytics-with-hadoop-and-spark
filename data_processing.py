from pyspark.ml.feature import VectorAssembler

def process_data(df):
    vector_assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
    df_transformed = vector_assembler.transform(df)
    return df_transformed
