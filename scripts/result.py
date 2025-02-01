
import pandas as pd


def load_time_response_result():
    usecols = ['Package','Name','Hyperparameters','Learning time']
    timeResponse = pd.read_csv("../data/results/timeResponse.csv", usecols=usecols)
    timeResponse=timeResponse.sort_values(["Package", "Name","Hyperparameters"])
    return timeResponse

def save_time_response_result(timeResponse): 
    timeResponse.to_csv('../data/results/timeResponse.csv', index=False) 

def update_time_response_result(package, name,hyperparameters, learningTime):
    learningTime=int(learningTime)
    #print(learningTime)
    timeResponsePandas = load_time_response_result()
    
    res=timeResponsePandas[(timeResponsePandas['Package']==package) 
        & (timeResponsePandas['Name']==name)
        & (timeResponsePandas['Hyperparameters']==hyperparameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        timeResponsePandas.loc[index, 'Learning time']=learningTime
    else:
        timeResponsePandas=pd.concat([pd.DataFrame([[package,name,hyperparameters,learningTime]], columns=timeResponsePandas.columns), timeResponsePandas], ignore_index=True)

    save_time_response_result(timeResponsePandas)