
import pandas as pd

def load_performance_test_result():
    usecols = ['Package','Name','Hyperparameters','F1','Mcc','ROC']
    data = pd.read_csv("../data/results/performancetest.csv", usecols=usecols)
    data=data.sort_values(["Package", "Name","Hyperparameters"])
    return data

def save_performance_test_result(timeResponse): 
    timeResponse.to_csv('../data/results/performancetest.csv', index=False) 

def update_performance_test_result(package, name,hyperparameters, f1,mcc,roc):
    f1=round(f1,4)
    mcc=round(mcc,4)
    roc=round(roc,4)
    data = load_performance_test_result()
    
    res=data[(data['Package']==package) 
        & (data['Name']==name)
        & (data['Hyperparameters']==hyperparameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        data.loc[index, 'F1']=f1
        data.loc[index, 'Mcc']=mcc
        data.loc[index, 'Roc']=roc
    else:
        data=pd.concat([pd.DataFrame([[package,name,hyperparameters,f1,mcc,roc]], columns=data.columns), data], ignore_index=True)

    save_performance_test_result(data)

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