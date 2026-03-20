import streamlit as st
import os
import requests
from requests.exceptions import HTTPError
import pandas as pd 

API_URL = os.getenv("API_URL","http://localhost:8000/")

def predict(row):
    try:
        payload = {
            "sepal_length": row["sepal length (cm)"],
            "sepal_width":  row["sepal width (cm)"],
            "petal_length": row["petal length (cm)"],
            "petal_width":  row["petal width (cm)"],
        }
        response = requests.post(url=f"{API_URL}predict", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None

def get_model_info():
    try:
        response = requests.get(f"{API_URL}model_infos")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Impossible de récupérer les infos du modèle : {e}")
        return None

model_info = get_model_info()
st.title("Iris Species Predictor")


if model_info:
    st.badge(f"Modèle v{model_info['model_infos']['version']}", icon="✅", color="green")
    
df = pd.read_csv("data/iris_test.csv", index_col=0).sample(n=10)
labels = ["Iris setosa","Iris versicolor","Iris virginica" ]
st.text("Charge une ligne de test pour faire une prediction ?")
charger = st.button("Charger")

if charger:
    sample = df.sample(n=1).iloc[0]
    st.write(sample.drop('target').to_frame().T)
    prediction_data = predict(sample)
    if prediction_data:
        label_index = prediction_data["prediction"]
        label = labels[label_index]
        version = prediction_data["version"]
        proba = prediction_data["probabilities"][label_index]
        st.success(f"Prédiction: {label} ({proba:.0%}) ")
    
    