import sys
import pandas as pd
import numpy as np
from pathlib import Path
from distances import haversine_distance,manhattan_distance,euclidean_distance

new_features = ['haversine_distance','euclidean_distance','manhattan_distance']

build_features_list = [haversine_distance,euclidean_distance,manhattan_distance]

def implement_distances(dataframe:pd.DataFrame, lat1:pd.Series,lon1:pd.Series,lat2:pd.Series,lon2:pd.Series) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for ind in range(len(build_features_list)):
        func = build_features_list[ind]
        dataframe[new_features[ind]] = func(lat1,lon1,lat2,lon2)
    
    return dataframe


def read_dataframe(path):
    df = pd.read_csv(path)
    return df


def save_dataframe(dataframe:pd.DataFrame, save_path:Path):
    dataframe.to_csv(save_path,index=False)


if __name__=="__main__":

    for ind in range(1,4):
        input_file_path = sys.argv[ind]
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        data_path = root_path/input_file_path
        filename = data_path.parts[-1]
        df = read_dataframe(data_path)
        df = implement_distances(dataframe=df,
                                 lat1=df['pickup_latitude'],
                                 lon1=df['pickup_longitude'],
                                 lat2=df['dropoff_latitude'],
                                 lon2=df['dropoff_longitude'])
        
        output_path = root_path/"data/processed/build-features"
        output_path.mkdir(exist_ok=True,parents=True)
        save_dataframe(df,output_path/filename)