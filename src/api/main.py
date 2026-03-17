from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
import os
from pydantic import BaseModel,  ConfigDict, ValidationError


# df = pd.read_csv("data/iris_test.csv", index_col=0)

app = FastAPI()
mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", "http://mlflow:5000"))
client = MlflowClient()
state = {
    "model": None,
    "version": None
}
MODEL_NAME = 'iris_model'
MODEL_ALIAS = 'production'

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    


def load_production_model():
    """Vérifie la version en production et recharge si nécessaire."""
    try:
        # On demande au Registry quelle est la version actuelle de l'alias 'Production'
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        prod_version = alias_info.version

        # Si le modèle n'est pas en cache ou si la version a changé sur MLflow
        if state["model"] is None or prod_version != state["version"]:
            print(f"🔄 Chargement de la version {prod_version} depuis MinIO...")
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            state["model"] = mlflow.pyfunc.load_model(model_uri)
            state["version"] = prod_version
           
        return state["model"]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur MLflow: {str(e)}")


@app.get("/")
async def root():
    return {"message" : "Hello World"}


@app.get("/model_infos")
def get_model_infos():
    load_production_model()
    return {"success": True, "model_infos": {"version": state["version"]}}


@app.post("/predict")
async  def predict(data: IrisData):
    input_df = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)":  data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)":  data.petal_width,
    }])
    # sample = df.sample(n=1, ignore_index=True)
    # sample = sample.drop(['target'], axis=1)
    model = load_production_model()
    if not model:
        return {"error" : "error loading model"}
    pred = model.predict(input_df)
    proba = model._model_impl.predict_proba(input_df)[0].tolist()

    return {"prediction": int(pred[0]), "probabilities": proba}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)