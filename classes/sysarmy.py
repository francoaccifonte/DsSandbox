import pandas as pd
import io
import requests
from time import sleep
from pathlib import Path
from sklearn.model_selection import train_test_split

csv_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTORpz5hc_wOwKdaKEIgVOAI3XVTh9WXB__C2abe_E0aJoYIoqhWQ-HTpBBryAByhIlX_booioqxK0T/pub?gid=1893349211&single=true&output=csv'
csv_path = Path.cwd() / 'dataset/sysarmy20201.csv'

class money20201():
    def __init__(self):
        self
    def perform(self):
        raw_dataframe = pd.read_csv(csv_path)
        columns_to_remove = [
            'Bolivia', 'Chile', 'Colombia', 'Cuba',
            'Costa Rica', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
            'México', 'Nicaragua', 'Panamá', 'Paraguay', 'Perú', 'Puerto Rico',
            'República Dominicana', 'Uruguay', 'Venezuela'
        ]
        raw_dataframe.drop(columns_to_remove, axis = 1, inplace = True)
        self.raw_dataframe = raw_dataframe
        pd.scatter_matrix



if __name__ == "__main__":
    money20201().perform()
