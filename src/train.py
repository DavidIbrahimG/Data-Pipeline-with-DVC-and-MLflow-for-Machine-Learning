import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os

# Train test split and grid search
from sklearn.model_selection import train_test_split, GridSearchCV

# MLFlow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/DavidIbrahimG/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "DavidIbrahimG"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "57784cd3735838b36046f55be85608fe82144f40"  # gotten under dagshub/remote/data/dvc/secret_access_key

# Function to run hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Load the train parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

# Train function
def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Set tracking URI
    mlflow.set_tracking_uri("https://dagshub.com/DavidIbrahimG/machinelearningpipeline.mlflow")

    # Start MLFlow run
    with mlflow.start_run():
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        signature = infer_signature(X_train, y_train)
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5], 
            'min_samples_leaf': [1, 2]
        }

        # Grid search
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Best model
        best_model = grid_search.best_estimator_

        # Predict with best model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of the best model on test data is: {accuracy}")

        # Log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        # Log confusion matrix and classification report into an artifact
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "confusion_report.txt")

        # Log the model **locally only**
        #mlflow.sklearn.log_model(best_model, "model", signature=signature)
        

        # Create directory to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the model using pickle (optional if you want a local copy too)
        filename = model_path
        pickle.dump(best_model, open(filename, 'wb'))

        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])
