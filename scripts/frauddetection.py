import warnings
warnings.filterwarnings('ignore')
import sourcedata as sd

import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.model_selection import train_test_split


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

def getPredictors(dataFrame):
    predictors = [col for col in dataFrame.columns ]
    predictors.remove('Class')
    predictors.remove('trx_date_time')
    return predictors

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
        return f1_score(y_test,y_pred)

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

def show_confusion_matrix(y_test,y_pred):
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
    plt.show()
