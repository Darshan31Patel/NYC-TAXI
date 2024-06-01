import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pathlib import Path
from sklearn.pipeline import Pipeline
from data_models import PredictionDataset
from src.features.distances import haversine_distance,manhattan_distance,euclidean_distance

app = FastAPI()

current_file_path = Path(__file__).parent
model_path = current_file_path / "models"/ "models" / "lgbm.joblib"
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
    lat1 = test_data.pickup_latitude
    lon1 = test_data.pickup_longitude
    lat2 = test_data.dropoff_latitude
    lon2 = test_data.dropoff_longitude

    distance_manhattan = manhattan_distance(lat1,lon1,lat2,lon2)
    distance_haversine = haversine_distance(lat1,lon1,lat2,lon2)
    distance_euclidean = euclidean_distance(lat1,lon1,lat2,lon2)

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
            'haversine_distance':distance_haversine,
            'euclidean_distance':distance_euclidean,
            'manhattan_distance':distance_manhattan
         }, index=[0]
    )

    predictions = model_pipeline.predict(X_test).reshape(-1,1)
    output = output_transformer.inverse_transform(predictions)[0].item()

    return f"Trip duration for the trip is {output:.2f} minutes"


if __name__=="__main__":
    uvicorn.run(app="app:app",port=8000)

