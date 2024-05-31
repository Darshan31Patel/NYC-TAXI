# import sys
# import joblib
# import pandas as pd
# from pathlib import Path
# from sklearn.metrics import r2_score

# TARGET = 'trip_duration'
# MODEL_NAME = 'rf.joblib'


# def load_dataframe(path):
#     df = pd.read_csv(path)
#     return df
    
    
# def make_X_y(dataframe:pd.DataFrame,target_column:str):
#     df_copy = dataframe.copy()
#     X = df_copy.drop(columns=target_column,axis=1)
#     y = df_copy[target_column]
    
#     return X, y


# def get_predictions(model, X:pd.DataFrame):
#     y_pred = model.predict(X)
#     return y_pred


# def calculate_r2_score(y_true,y_pred):
#     score = r2_score(y_true=y_true,y_pred=y_pred)
#     return score


# def main():
#     current_path = Path(__file__)
#     root_path = current_path.parent.parent.parent

#     for ind in range(1,3):
#         data_path = root_path/'data/processed/final'/sys.argv[ind]
#         data = load_dataframe(data_path)
#         X_test,y_test = make_X_y(dataframe=data,target_column=TARGET)
#         model_path = root_path/'models'/'models'/MODEL_NAME

#         model = joblib.load(model_path)
#         y_pred = get_predictions(model=model,X=X_test)
#         score = calculate_r2_score(y_true=y_test,y_pred=y_pred)

#         print(f"\n The score for dataset {sys.argv[ind]} is {score}")

# if __name__ == "__main__":
#     main()



import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

TARGET = 'trip_duration'
MODEL_NAME = 'rf.joblib'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df

def make_X_y(dataframe: pd.DataFrame, target_column: str):
    df_copy = dataframe.copy()
    X = df_copy.drop(columns=target_column, axis=1)
    y = df_copy[target_column]
    return X, y

def get_predictions(model, X: pd.DataFrame):
    y_pred = model.predict(X)
    return y_pred

def calculate_r2_score(y_true, y_pred):
    score = r2_score(y_true=y_true, y_pred=y_pred)
    return score

def log_mlflow(params, model, model_name, data_path=None, score=None):
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        if data_path:
            mlflow.log_param("data_path", str(data_path))
        if score is not None:
            mlflow.log_metric("r2_score", score)
        if model:
            mlflow.sklearn.log_model(model, model_name)

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent

    for ind in range(1, 3):
        data_path = root_path / 'data/processed/final' / sys.argv[ind]
        data = load_dataframe(data_path)
        X_test, y_test = make_X_y(dataframe=data, target_column=TARGET)
        model_path = root_path / 'models' / 'models' / MODEL_NAME

        model = joblib.load(model_path)
        y_pred = get_predictions(model=model, X=X_test)
        score = calculate_r2_score(y_true=y_test, y_pred=y_pred)

        print(f"\nThe score for dataset {sys.argv[ind]} is {score}")

        # Use the MLflow logging function
        log_mlflow(
            params=None, 
            model=model, 
            model_name="model", 
            data_path=data_path, 
            score=score
        )

if __name__ == "__main__":
    main()
