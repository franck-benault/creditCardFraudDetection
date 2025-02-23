import warnings
warnings.filterwarnings('ignore')
import sourcedata as sd

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def getPredictors(dataFrame):
    predictors = [col for col in dataFrame.columns ]
    predictors.remove('Class')
    predictors.remove('db_uuid')
    predictors.remove('card_pan_id')
    predictors.remove('term_mcc')
    predictors.remove('term_country')
    predictors.remove('trx_date_time')
    predictors.remove('clusterCardHolder_UNKNOWN')
    predictors.remove('clusterMerchant_UNKNOWN')
    return predictors

#calculate information value 
def calc_iv(df, feature, target, pr=0):
    lst = []
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])
    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
    data = data[data['Bad'] > 0]
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()
    data = data.sort_values(by=['Variable', 'Value'], ascending=True)
    #print(data)
    if pr == 1:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Value', y='WoE', data=data)
        ax.set_title('WOE visualization for: ' )
        plt.show()
        print(data)
    return data['IV'].values[0]
#calc_iv(amazon_turk,“age_bin”,“Y”,pr=0)

def getAllFiles():
    files =['export20241118.csv','export20241119.csv','export20241120.csv','export20241121.csv','export20241125.csv']
    return files


def split_data(data_df, predictors, target='Class', scaler=None):
    TEST_SIZE = 0.20 # test size using_train_test_split
    RANDOM_STATE = 0
    columns=predictors.copy()
    columns.append(target)
    #print(columns)
    df = data_df[columns]

    x_train, x_test, y_train, y_test = train_test_split(data_df[predictors], data_df[target], test_size = TEST_SIZE, 
                                                        stratify=data_df[target],
                                                        random_state = RANDOM_STATE)
    #print("fraudulent transactions rate per mil full  data", 1000*data_df[data_df['Class'] == 1].shape[0]/data_df.shape[0])
    #print("fraudulent transactions rate per mil train data", 1000*y_train[y_train == 1].shape[0]/y_train.shape[0])
    #print("fraudulent transactions rate per mil test  data", 1000*y_test[y_test == 1].shape[0]/y_test.shape[0])

    if(scaler==None):
        return x_train, x_test, y_train, y_test, None
    else:
        x_train= scaler.fit_transform(x_train) 
        x_test= scaler.transform(x_test)
        x_train = pd.DataFrame(x_train, columns=[predictors])
        x_test = pd.DataFrame(x_test, columns=[predictors])
        return x_train, x_test, y_train, y_test, scaler


def print_scores(y_test,y_pred,scoreType,drawRocCurve=False):
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    #print('classification report :\n', classification_report(y_test,y_pred)) 
    print("accuracy score: {:.4f}".format(accuracy_score(y_test,y_pred)))
    print("balanced accuracy score: {:.4f}".format(balanced_accuracy_score(y_test,y_pred)))
    print("recall score: {:.4f}".format(recall_score(y_test,y_pred)))
    print("precision score: {:.4f}".format(precision_score(y_test,y_pred)))
    print("f1 score: {:.4f}".format(f1_score(y_test,y_pred)))  
    print("mcc score: {:.4f}".format(matthews_corrcoef(y_test,y_pred)))
    
    print("roc auc score: {:.4f}".format(roc_auc_score(y_test,y_pred)))
    if(drawRocCurve):
        draw_roc_curve(y_test,y_pred)

    if(scoreType=='f1'):
        return f1_score(y_test,y_pred)
    if(scoreType=='accuraty'):
        return accuracy_score(y_test,y_pred)
    if(scoreType=='recall'):
        return recall_score(y_test,y_pred)
    if(scoreType=='roc_auc_score'):
        return roc_auc_score(y_test,y_pred)
    if(scoreType=='precision'):
        return precision_score(y_test,y_pred)
    else:
        return f1_score(y_test,y_pred),matthews_corrcoef(y_test,y_pred),roc_auc_score(y_test,y_pred)

def show_importance(modelClf,predictors):
    if(hasattr(modelClf,"feature_importances_")):
        tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': modelClf.feature_importances_})
        tmp = tmp.sort_values(by='Feature importance',ascending=False)
        plt.figure(figsize = (7,4))
        plt.title('Features importance',fontsize=14)
        s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
        plt.show() 
    else:
        print("No feature importance")

def show_confusion_matrix(y_test,y_pred,imageName=None):
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
    sns.heatmap(cm, 
    xticklabels=['Not Fraud', 'Fraud'],
    yticklabels=['Not Fraud', 'Fraud'],
    annot=True,ax=ax1,
    fmt='d',
    linewidths=.2,linecolor="Darkblue", cmap="Blues")
    cm.style.format("{:20}")
    plt.title('Confusion Matrix', fontsize=14)
    if(imageName!=None):
        plt.savefig(imageName)
    plt.show()

def show_prediction_graph(modelClf, x_test,y_test,imageName=None):
    prediction=modelClf.predict_proba(x_test)[:,1]
    plt.figure(figsize=(10,5))
    list =np.array([])
    list0=prediction[y_test==0]
    list1=prediction[y_test==1]
    plt.hist(list0, bins=20, label='Negatives',alpha=0.5)
    for i in np.arange(0,len(list0)/len(list1)): 
        list= np.append(list,list1)

    plt.hist(list, bins=20, label='Positives', alpha=0.5, color='r')
    plt.xlabel('Probability of being Positive Class', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    if(imageName!=None):
        plt.savefig(imageName)
    plt.show() 

def draw_roc_curve(y_test,y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred) 
    roc_auc = roc_auc_score(y_test, y_pred) 
    # Plot the ROC curve
    plt.figure()  
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Classification')
    plt.legend()
    plt.show()

def plt_train_test(range, tabf1Train,tabf1Test=[]):
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot()

    ax1.set_ylabel("f1 Train")
    ax1.plot(range, tabf1Train, color = 'red', label = 'f1 Train')
    ax1.legend(loc = 'upper left')

    if(len(tabf1Test)==len(tabf1Train)):
        ax2 = ax1.twinx()
        ax2.set_ylabel("f1 test")
        ax2.plot(range, tabf1Test, color = 'blue', label = 'f1 Test')
        ax2.legend(loc = 'upper right')

    fig.autofmt_xdate()
    plt.show()

def calculate_scores(y_test,y_pred,scoreType):
    if(scoreType=='f1'):
        return f1_score(y_test,y_pred)
    if(scoreType=='accuraty'):
        return accuracy_score(y_test,y_pred)
    if(scoreType=='recall'):
        return recall_score(y_test,y_pred)
    if(scoreType=='roc_auc_score'):
        return roc_auc_score(y_test,y_pred)
    if(scoreType=='precision'):
        return precision_score(y_test,y_pred)
    else:
        return f1_score(y_test,y_pred)

def getScalers():
    scalers={"StandardScaler":StandardScaler(),
            "MinMaxScaler":MinMaxScaler(), 
            "RobustScaler":RobustScaler(),
            "MaxAbsScaler":MaxAbsScaler()}
    return scalers

def processModel(model, dfTrx, predictors, drop_list,
                 parameters={"max_iter":1000},scaler=None):  
    then= datetime.now()
    #print(parameters)
    model.set_params(**parameters)
    
    for y in drop_list:
        #print(y)
        predictors.remove(y)

    then= datetime.now()
    x_train, x_test, y_train, y_test, scaler  =split_data(dfTrx,predictors, 'Class',scaler)
    model.fit(x_train, y_train)
    predsTrain = model.predict(x_train)
    predsTest = model.predict(x_test)

    scoreType="f1"
    f1Train = calculate_scores(y_train,predsTrain,scoreType)
    f1Test  = calculate_scores(y_test,predsTest,scoreType)

    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print(f"Duration {duration_in_s:.1f} and now {now}")
    print(f"duration_in_s {duration_in_s:.1f} parameters {parameters} scaler {scaler} f1Train {f1Train:.4f} f1Test {f1Test:.4f}")
    return duration_in_s,f1Train,f1Test, scaler

def hyperparameterSelectionGridSearchCV(classifier, dic_param, scoring, dfTrxEncoded2, predictors, drop_list,scaler):
    for y in drop_list:
        #print(y)
        predictors.remove(y)

    x_train, x_test, y_train, y_test, scaler = split_data(dfTrxEncoded2,predictors, 'Class',scaler)
    grid = GridSearchCV(classifier,dic_param, scoring=scoring, verbose=10,cv=2).fit(x_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)
    
    y_pred=grid.predict(x_train)
    scoref1=calculate_scores(y_train,y_pred,'f1')
    print("scoref1",scoref1)
    show_confusion_matrix(y_pred, y_train)
    
    return grid.best_params_

def hyperparameterSelectionRandomizedSearchCV(classifier, dic_param, scoring, dfTrxEncoded2, predictors, drop_list, scaler):
    for y in drop_list:
        #print(y)
        predictors.remove(y)

    x_train, x_test, y_train, y_test, scaler = split_data(dfTrxEncoded2,predictors, 'Class',scaler)
    random_search = RandomizedSearchCV(classifier,dic_param, scoring=scoring, verbose=10,cv=2,n_iter=10).fit(x_train, y_train)
    print(random_search.best_params_)
    print(random_search.best_score_)
    score=random_search.score(x_train,y_train)
    y_pred=random_search.predict(x_train)
    scoref1=calculate_scores(y_train,y_pred,'f1')
    print("score  ",score)
    print("scoref1",scoref1)
    show_confusion_matrix(y_pred, y_train)
    return random_search.best_params_