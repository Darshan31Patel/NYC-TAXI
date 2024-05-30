import sys
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from src.logger import CustomLogger,create_log_path

TARGET_COLUMN = 'trip_duration'
PLOT_PATH = Path('reports/figures/target_distribution.png')

log_file_path = create_log_path('modify_features')
modify_logger = CustomLogger(logger_name='modify_features',log_filename=log_file_path)
modify_logger.set_log_level(level=logging.INFO)


def convert_target_to_minutes(dataframe:pd.DataFrame, target_column:str) -> pd.DataFrame:
    # dataframe.loc[:,target_column] = dataframe[target_column]/60
    dataframe.loc[:, target_column] = (dataframe[target_column] / 60).astype(float)
    modify_logger.save_logs(msg='Target column converted from seconds into minutes')
    return dataframe


def drop_above_two_hundred_minutes(dataframe:pd.DataFrame, target_column:str) -> pd.DataFrame:
    # filter rows having target less than 200
    filter_series = dataframe[target_column]<=200
    new_dataframe = dataframe.loc[filter_series,:].copy()

    # max value of target to check the outliers are removed
    max_value = new_dataframe[target_column].max()
    modify_logger.save_logs(msg=f'The max value in target columnn after transformation is {max_value} and the state of transformation is {max_value<=200}')
    if max_value <=200:
        return dataframe
    else:
        raise ValueError("outliers target values not removed from data")
    

def plot_target(dataframe:pd.DataFrame, target_column:str, save_path:Path):
    sns.kdeplot(data=dataframe,x=target_column)
    plt.title(f'Distribution of {target_column}')
    plt.savefig(save_path)
    modify_logger.save_logs(msg='Distribution plot saved at destination')


def drop_columns(dataframe:pd.DataFrame) -> pd.DataFrame:
    modify_logger.save_logs(msg=f'columns in data before removal are {list(dataframe.columns)}')

    if 'dropoff_datetime' in dataframe.columns:
        columns_to_drop = ['id','dropoff_datetime','store_and_fwd_flag']
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)

        modify_logger.save_logs(msg=f'columns after removal are {list(dataframe_after_removal.columns)}')

        return dataframe_after_removal
    
    else:
        columns_to_drop = ['id','store_and_fwd_flag']
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)

        modify_logger.save_logs(msg=f'columns after removal are {list(dataframe_after_removal.columns)}')

        return dataframe_after_removal
    

def make_datetime_features(dataframe:pd.DataFrame) -> pd.DataFrame:
    new_dataframe = dataframe.copy()
    original_number_rows, original_number_columns = new_dataframe.shape

    new_dataframe['pickup_datetime'] = pd.to_datetime(new_dataframe['pickup_datetime'])
    modify_logger.save_logs(msg=f"pickup_datetime converted to datetime {new_dataframe['pickup_datetime'].dtype}")

    new_dataframe.loc[:,'pickup_hour'] = new_dataframe['pickup_datetime'].dt.hour 
    new_dataframe.loc[:,'pickup_date'] = new_dataframe['pickup_datetime'].dt.day
    new_dataframe.loc[:,'pickup_month'] = new_dataframe['pickup_datetime'].dt.month
    new_dataframe.loc[:,'pickup_day'] = new_dataframe['pickup_datetime'].dt.weekday
    new_dataframe.loc[:,'is_weekend'] = new_dataframe.apply(lambda row: row['pickup_day'] >= 5,axis=1).astype('int')

    new_dataframe = new_dataframe.drop(columns=['pickup_datetime'])
    modify_logger.save_logs(msg="pickup_datetime column dropped")

    transformed_number_row, transformed_number_column = new_dataframe.shape
    modify_logger.save_logs(msg=f'The number of columns increased by 4 {transformed_number_column == (original_number_columns + 5 - 1)}')
    modify_logger.save_logs(msg=f'The number of rows remained the same {original_number_rows == transformed_number_row}')

    return new_dataframe


def remove_passengers(dataframe:pd.DataFrame) -> pd.DataFrame:
    passenger_to_include = list(range(1,7))
    new_dataframe_filter = dataframe['passenger_count'].isin(passenger_to_include)

    new_dataframe = dataframe.loc[new_dataframe_filter,:]
    unique_passenger_values = list(np.sort(new_dataframe['passenger_count'].unique()))

    modify_logger.save_logs(msg=f'the unique passenger list is {unique_passenger_values} verify={passenger_to_include == unique_passenger_values}')

    return new_dataframe


def input_modifications(dataframe:pd.DataFrame) -> pd.DataFrame:
    new_dataframe = drop_columns(dataframe=dataframe)
    df_passengers_modifications = remove_passengers(new_dataframe)
    df_with_datetime_features = make_datetime_features(df_passengers_modifications)
    modify_logger.save_logs(msg='modification with input features complete')

    return df_with_datetime_features


def target_modifications(dataframe:pd.DataFrame, target_column:str = TARGET_COLUMN) -> pd.DataFrame:
    minutes_dataframe = convert_target_to_minutes(dataframe,target_column)
    target_outliers_removed_df = drop_above_two_hundred_minutes(minutes_dataframe,target_column)
    plot_target(dataframe=target_outliers_removed_df,target_column=target_column,save_path=PLOT_PATH)
    modify_logger.save_logs('Modification with target feature complete')

    return target_outliers_removed_df


def read_data(data_path):
    df = pd.read_csv(data_path)
    return df

def save_data(dataframe:pd.DataFrame,save_path:Path):
    dataframe.to_csv(save_path)


def main(data_path,filename):
    df = read_data(data_path)
    df_input_modification = input_modifications(df)

    if (filename=='train.csv') or (filename=='val.csv'):
        df_final = target_modifications(df_input_modification)
    else:
        df_final = df_input_modification
    
    return df_final



if __name__=="__main__":
    # for train test val dataset
    for ind in range(1,4):
        input_file_path = sys.argv[ind]
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        data_path = root_path/input_file_path
        filename = data_path.parts[-1]

        df_final = main(data_path=data_path,filename=filename)
        
        output_path = root_path/'data/processed/transformations'
        output_path.mkdir(parents=True,exist_ok=True)

        save_data(df_final,output_path/filename)
        modify_logger.save_logs(msg=f'{filename} saved at destination folder')