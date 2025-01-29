
import pandas as pd

def load_time_response_result():
    usecols = ['Package','Name','Hyperparamters','Learning time']
    timeResponse = pd.read_csv("../data/results/timeResponse.csv", usecols=usecols)
    return timeResponse

def save_time_response_result(timeResponse): 
    timeResponse.to_csv('../data/results/timeResponse.csv', index=False) 