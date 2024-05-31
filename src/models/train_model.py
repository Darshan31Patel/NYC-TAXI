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

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    training_data_path = root_path / sys.argv[1]
    train_data = load_dataframe(training_data_path)
    X_train, y_train = make_X_y(dataframe=train_data, target_column=TARGET)

    with open("params.yaml") as f:
        params = safe_load(f)

    model_params = params['train_model']['random_forest_regressor']

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(model_params)

        regressor = RandomForestRegressor(**model_params)
        regressor = train_model(model=regressor, X_train=X_train, y_train=y_train)

        model_output_path = root_path / 'models' / 'models'
        model_output_path.mkdir(exist_ok=True)

        model_save_path = model_output_path / 'rf.joblib'
        save_model(model=regressor, save_path=model_save_path)

        # Log model
        mlflow.sklearn.log_model(regressor, "random_forest_model")

        # Log the training data path
        mlflow.log_param("training_data_path", training_data_path)

        # If there are metrics to log, you can log them here
        # e.g., mlflow.log_metric("example_metric", example_metric_value)

        # Log the model file as an artifact
        mlflow.log_artifact(model_save_path)

if __name__ == "__main__":
    main()
