from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv( "MINIO_ROOT_USER")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv( "MINIO_ROOT_PASSWORD")

def prepare_minio():
    """Vérifie si le bucket 'mlflow' existe, sinon le crée"""


    s3 = boto3.client('s3', endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'])


    buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]


    if 'mlflow' not in buckets:
        s3.create_bucket(Bucket='mlflow')
        print("Bucket 'mlflow' créé avec succès.")


def train_and_register():
    # Configuration du serveur de tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("My_Experiment")
    # Chargement des données train/test

    with mlflow.start_run() as run:
        # Entraînement
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "test_size": 0.2,
            "random_state": 42,
            "scaler": "StandardScaler",
            # "model": "LogisticRegression",
            "model": "RandomForestClassifier",
        }

        model = Pipeline([
            ('scaler', StandardScaler()),
            # ('classifier', LogisticRegression())
            ('classifier', RandomForestClassifier())
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Log des paramètres et metrics
        mlflow.log_params(params)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred,  average='weighted')
        f1 = f1_score(y_test, y_pred , average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        # 2. Enregistrement dans MinIO ET dans le Model Registry
        # On définit le nom du modèle dans le catalogue
        model_name = "iris_model"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )      
    # 3. Gestion de l'Alias 'Production' via MlflowClient
    client = MlflowClient()


    # On récupère la toute dernière version créée
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version


    # On lui attribue l'alias 'Production'
    # client.set_registered_model_alias(model_name, "Production", latest_version)

prepare_minio()
train_and_register()

