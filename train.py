import os
os.makedirs("artifacts", exist_ok=True)
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

df = fetch_california_housing(as_frame=True).frame
X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
pipe.fit(X_tr, y_tr)
print("R2:", pipe.score(X_te, y_te))
joblib.dump(pipe, "artifacts/pipeline.joblib")
X_te.head(5).to_csv("artifacts/sample_input.csv", index=False)
