import joblib, pandas as pd
pipe = joblib.load("artifacts/pipeline.joblib")
print(pipe.predict(pd.read_csv("artifacts/sample_input.csv")))

