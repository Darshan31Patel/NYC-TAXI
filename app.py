import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pathlib import Path
from sklearn.pipeline import Pipeline
from data_models import PredictionDataset

app = FastAPI()

current_file_path = Path(__file__).parent
model_path = current_file_path / "models"/ "models" / "rf.joblib"
preprocessor_path = model_path.parent.parent / "transformers" / "preprocessor.joblib"
output_transformer_path = preprocessor_path.parent / "output_transformer.joblib"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
output_transformer = joblib.load(output_transformer_path)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ('regressor',model)
])

@app.get('/')
def home():
    return "Welcome NYC Taxi Duration Prediction App"


@app.post('/predictions')
def predictions(test_data:PredictionDataset):
    X_test = pd.DataFrame(
        data = {
            'vendor_id':test_data.vendor_id,
            'passenger_count':test_data.passenger_count,
            'pickup_longitude':test_data.pickup_longitude,
            'pickup_latitude':test_data.pickup_latitude,
            'dropoff_longitude':test_data.dropoff_longitude,
            'dropoff_latitude':test_data.dropoff_latitude,
            'pickup_hour':test_data.pickup_hour,
            'pickup_date':test_data.pickup_date,
            'pickup_month':test_data.pickup_month,
            'pickup_day':test_data.pickup_day,
            'is_weekend':test_data.is_weekend,
            'haversine_distance':test_data.haversine_distance,
            'euclidean_distance':test_data.euclidean_distance,
            'manhattan_distance':test_data.manhattan_distance
         }, index=[0]
    )

    predictions = model_pipeline.predict(X_test).reshape(-1,1)
    output = output_transformer.inverse_transform(predictions)[0].item()

    return f"Trip duration for the trip is {output:.2f} minutes"


if __name__=="__main__":
    uvicorn.run(app="app:app",port=8000)