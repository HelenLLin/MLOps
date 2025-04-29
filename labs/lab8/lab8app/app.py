from fastapi import FastAPI, Request
import pandas as pd
import mlflow.sklearn
import uvicorn

app = FastAPI(title="Life Expectancy API")

mlflow.set_tracking_uri("http://127.0.0.1:5001/")
mlflow.set_experiment("lab8app-experiment")
model = f"runs:/97f7cde8c33943198ae490f8672f5cde/final_model"
loaded_model = mlflow.sklearn.load_model(model)

features = [
    'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
    'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio',
    'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
    'thinness  1-19 years', 'thinness 5-9 years',
    'Income composition of resources', 'Schooling',
    'Status_Developed', 'Status_Developing'
]

@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
        input_data = payload.get("data")
        if not input_data:
            return {"error": "No input data provided"}

        input_df = pd.DataFrame(input_data, columns=features)
        prediction = loaded_model.predict(input_df)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}