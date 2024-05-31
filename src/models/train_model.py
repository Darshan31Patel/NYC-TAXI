# import sys
# import joblib
# import pandas as pd
# from yaml import safe_load
# from pathlib import Path
# from sklearn.ensemble import RandomForestRegressor

# TARGET = 'trip_duration'


# def load_dataframe(path):
#     df = pd.read_csv(path)
#     return df


# def make_X_y(dataframe:pd.DataFrame, target_column:str):
#     df_copy =  dataframe.copy()
#     X = df_copy.drop(columns=target_column)
#     y = df_copy[target_column]

#     return X,y


# def train_model(model,X_train,y_train):
#     model.fit(X_train,y_train)
#     return model


# def save_model(model,save_path):
#     joblib.dump(model,save_path)


# def main():
#     current_path = Path(__file__)
#     root_path = current_path.parent.parent.parent
#     training_data_path = root_path/sys.argv[1]
#     train_data = load_dataframe(training_data_path)
#     X_train,y_train = make_X_y(dataframe=train_data,target_column=TARGET)

#     with open("params.yaml") as f:
#         params = safe_load(f)

#     model_params = params['train_model']['random_forest_regressor']

#     regressor = RandomForestRegressor(**model_params)
#     regressor = train_model(model=regressor,X_train=X_train,y_train=y_train)

#     model_output_path = root_path/'models'/'models'
#     model_output_path.mkdir(exist_ok=True)

#     save_model(model=regressor, save_path=model_output_path/'rf.joblib')


# if __name__ == "__main__":
#     main()



import sys
import joblib
import pandas as pd
from yaml import safe_load
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

TARGET = 'trip_duration'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df

def make_X_y(dataframe: pd.DataFrame, target_column: str):
    df_copy = dataframe.copy()
    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]
    return X, y

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def save_model(model, save_path):
    joblib.dump(model, save_path)

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
    training_data_path = root_path / sys.argv[1]
    train_data = load_dataframe(training_data_path)
    X_train, y_train = make_X_y(dataframe=train_data, target_column=TARGET)

    with open("params.yaml") as f:
        params = safe_load(f)

    model_params = params['train_model']['random_forest_regressor']

    regressor = RandomForestRegressor(**model_params)
    regressor = train_model(model=regressor, X_train=X_train, y_train=y_train)

    model_output_path = root_path / 'models' / 'models'
    model_output_path.mkdir(exist_ok=True)

    model_save_path = model_output_path / 'rf.joblib'
    save_model(model=regressor, save_path=model_save_path)

    # Use the MLflow logging function
    log_mlflow(
        params=model_params, 
        model=regressor, 
        model_name="random_forest_model", 
        data_path=training_data_path
    )

if __name__ == "__main__":
    main()
