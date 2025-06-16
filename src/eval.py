import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

# MLFlow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/DavidIbrahimG/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "DavidIbrahimG"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "57784cd3735838b36046f55be85608fe82144f40"  # gotten under dagshub/remote/data/dvc/secret_access_key

# load parameters from param.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    ## Set tracking uri 
    mlflow.set_tracking_uri("https://dagshub.com/DavidIbrahimG/machinelearningpipeline.mlflow")

    ## load the model from the file path
    model = pickle.load(open(model_path, 'rb'))

    ## prediction
    prediction = model.predict(X)
    accuracy = accuracy_score(y, prediction)

    ## log the model metrics
    mlflow.log_metric("accuracy", accuracy)

    ## print model accuracy
    print(f"your model accuracy is:{accuracy}")


if __name__  == "__main__":
    evaluate(params["data"], params["model"])
