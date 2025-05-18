
import pandas as pd
import numpy as np

def update_hyperparameter_config_result(package, name, extraparameters,max_depth, n_estimators=np.nan):
    data = load_hyperparameter_config_result()
    
    res=data[(data['Package']==package) 
        & (data['Name']==name)
        & (data['ExtraParameters']==extraparameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        data.loc[index, 'max_depth']=max_depth
        data.loc[index, 'n_estimators']=n_estimators
    else:
        data=pd.concat([pd.DataFrame([[package, name, extraparameters,max_depth,n_estimators]], columns=data.columns), data], ignore_index=True)

    save_hyperparameter_config_result(data)

def save_hyperparameter_config_result(data):
    data.to_csv('../data/results/hyperparameterConfigResult.csv', index=False)   

def load_hyperparameter_config_result():
    usecols = ['Package','Name','ExtraParameters','max_depth','n_estimators']
    data = pd.read_csv("../data/results/hyperparameterConfigResult.csv", usecols=usecols)
    data=data.sort_values(['max_depth'])
    return data

def load_features_IV_result():
    usecols = ['Feature','IV']
    data = pd.read_csv("../data/results/featureIV.csv", usecols=usecols)
    data=data.sort_values(['IV'])
    return data

def save_features_IV_result(data): 
    data.to_csv('../data/results/featureIV.csv', index=False) 

def update_features_IV_result(feature, iv):
    iv=round(iv,4)
    #print(iv)
    data = load_features_IV_result()
    
    res=data[(data['Feature']==feature)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        data.loc[index, 'IV']=iv
    else:
        data=pd.concat([pd.DataFrame([[feature,iv]], columns=data.columns), data], ignore_index=True)

    save_features_IV_result(data)

def load_performance_nextdays_result():
    usecols = ['Package','Name','ExtraParameters','F1 Day1','F1 Day2','F1 Day3','F1 Day4','ROC Day1','ROC Day2','ROC Day3','ROC Day4']
    data = pd.read_csv("../data/results/performanceNextDays.csv", usecols=usecols)
    data=data.sort_values(["Package", "Name","ExtraParameters"])
    return data

def save_performance_nextdays_result(timeResponse): 
    timeResponse.to_csv('../data/results/performanceNextDays.csv', index=False) 

def update_performance_nextdays_result(package, name,extraParameters, f1Day1,f1Day2,f1Day3,f1Day4,rocDay1,rocDay2,rocDay3,rocDay4):
    f1Day1=round(f1Day1,4)
    f1Day2=round(f1Day2,4)
    f1Day3=round(f1Day3,4)
    f1Day4=round(f1Day4,4)

    rocDay1=round(rocDay1,4)
    rocDay2=round(rocDay2,4)
    rocDay3=round(rocDay3,4)
    rocDay4=round(rocDay4,4)

    data = load_performance_nextdays_result()
    
    res=data[(data['Package']==package) 
        & (data['Name']==name)
        & (data['ExtraParameters']==extraParameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        data.loc[index, 'F1 Day1']=f1Day1
        data.loc[index, 'F1 Day2']=f1Day2
        data.loc[index, 'F1 Day3']=f1Day3
        data.loc[index, 'F1 Day4']=f1Day4

        data.loc[index, 'ROC Day1']=rocDay1
        data.loc[index, 'ROC Day2']=rocDay2
        data.loc[index, 'ROC Day3']=rocDay3
        data.loc[index, 'ROC Day4']=rocDay4

    else:
        data=pd.concat([pd.DataFrame([[package,name,extraParameters,f1Day1,f1Day2,f1Day3,f1Day4,rocDay1,rocDay2,rocDay3,rocDay4]], columns=data.columns), data], ignore_index=True)

    save_performance_nextdays_result(data)

def load_performance_test_result():
    usecols = ['Package','Name','ExtraParameters','F1','Mcc','ROC']
    data = pd.read_csv("../data/results/performancetest.csv", usecols=usecols)
    data=data.sort_values(["Package", "Name","ExtraParameters"])
    return data

def save_performance_test_result(timeResponse): 
    timeResponse.to_csv('../data/results/performancetest.csv', index=False) 

def update_performance_test_result(package, name,extraParameters, f1,mcc,roc):
    f1=round(f1,4)
    mcc=round(mcc,4)
    roc=round(roc,4)
    data = load_performance_test_result()
    
    res=data[(data['Package']==package) 
        & (data['Name']==name)
        & (data['ExtraParameters']==extraParameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        data.loc[index, 'F1']=f1
        data.loc[index, 'Mcc']=mcc
        data.loc[index, 'ROC']=roc
    else:
        data=pd.concat([pd.DataFrame([[package,name,extraParameters,f1,mcc,roc]], columns=data.columns), data], ignore_index=True)

    save_performance_test_result(data)

def load_time_response_result():
    usecols = ['Package','Name','ExtraParameters','Learning time']
    timeResponse = pd.read_csv("../data/results/timeResponse.csv", usecols=usecols)
    timeResponse=timeResponse.sort_values(["Package", "Name","ExtraParameters"])
    return timeResponse

def save_time_response_result(timeResponse): 
    timeResponse.to_csv('../data/results/timeResponse.csv', index=False) 

def update_time_response_result(package, name,extraParameters, learningTime):
    learningTime=int(learningTime)
    #print(learningTime)
    timeResponsePandas = load_time_response_result()
    
    res=timeResponsePandas[(timeResponsePandas['Package']==package) 
        & (timeResponsePandas['Name']==name)
        & (timeResponsePandas['ExtraParameters']==extraParameters)]
    #print(res.shape[0])
    if (res.shape[0]>0):
        index=res.index[0]
        #print('trace')
        timeResponsePandas.loc[index, 'Learning time']=learningTime
    else:
        timeResponsePandas=pd.concat([pd.DataFrame([[package,name,extraParameters,learningTime]], columns=timeResponsePandas.columns), timeResponsePandas], ignore_index=True)

    save_time_response_result(timeResponsePandas)