import sys
import os
import datetime

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from io import BytesIO
from sklearn.metrics import accuracy_score
import pickle
import joblib

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pprint import pprint


#pathname = os.path.dirname(sys.argv[0])
#path = os.path.abspath(pathname)

# Load Iris dataset
data1 = pd.read_csv("data/data.csv")

# Create Test Train Data
train, test = train_test_split(
    data1, test_size=0.4, stratify=data1["species"], random_state=42
)
X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_train = train.species
X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_test = test.species


# Create Model using params 
params = {
        "max_depth" :3 ,
        "random_state" :1
}

mod_dt = DecisionTreeClassifier(**params)
mod_dt.fit(X_train, y_train)
prediction = mod_dt.predict(X_test)
accuracyOfModel = metrics.accuracy_score(prediction, y_test)
print(
    "The accuracy of the Decision Tree is",
    "{:.3f}".format(accuracyOfModel),
)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Dump model using joblib locally
joblib.dump(mod_dt, "artifacts/model.joblib")
print(" model saved")

# Save metrics.csv
metrics_df = pd.DataFrame({"accuracy": [accuracyOfModel]})
metrics_df.to_csv("metrics.csv", index=False)
print(" metrics saved")


# Setup MLFLOW and save model and experiment runs 
mlflow.set_tracking_uri("http://34.55.225.41:8100/")
client = MlflowClient(mlflow.get_tracking_uri())
all_experiments = client.search_experiments()
print(all_experiments)

print(mlflow.get_tracking_uri())

mlflow.set_experiment("IRIS Classifier: Test")



with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy",accuracyOfModel)
    mlflow.set_tag("Training Info", "IRIS Classifier:Decision Tree : Mlflow")
    modelsignature = infer_signature(X_train,mod_dt.predict(X_train))
    mlflow.sklearn.log_model(sk_model = mod_dt, artifact_path="Iris model",signature = modelsignature , input_example = X_train , registered_model_name="IRIS-classifier-dt_v1")




