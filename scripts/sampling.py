from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
#from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import OneSidedSelection

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pickle as pkl

def saveTrainData(x_train,y_train,fileName):
    with open("../data/cleaned/"+fileName, "wb") as f:
        pkl.dump([x_train, y_train], f)

def loadTrainData(fileName):
    with open("../data/cleaned/"+fileName, "rb") as f:
        train_x, train_y = pkl.load(f)
    return train_x.copy(),train_y.copy()


def draw2DPCAScaterPlotV2(xTrain,yTrain,xTrain2,yTrain2,title='After sampling'):
    pca = PCA(n_components=2)
    sc = StandardScaler()
    x2 = sc.fit_transform(xTrain)
    x_pca = pca.fit_transform(x2)

    # giving a larger plot
    fig,ax = plt.subplots(1,2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    ax[0].set_title('Before sampling')
    ax[0].scatter(x_pca[:, 0], x_pca[:, 1],
            c=yTrain,
            cmap='plasma')
    # labeling x and y axes
    ax[0].set_xlabel('First Principal Component')
    ax[0].set_ylabel('Second Principal Component')

    x2 = sc.transform(xTrain2)
    x_pca = pca.transform(x2)

    ax[1].set_title(title)
    ax[1].scatter(x_pca[:, 0], x_pca[:, 1],
            c=yTrain2,
            cmap='plasma')

    # labeling x and y axes
    ax[1].set_xlabel('First Principal Component')
    ax[1].set_ylabel('Second Principal Component')

    plt.show()

def randomUnderSampling(x,y,rate=1/100):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))

    then= datetime.now()
    undersample = RandomUnderSampler(sampling_strategy=rate,random_state=42)
    x_train, y_train = undersample.fit_resample(x, y)
    
    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))
    
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train

def nearMissV1UnderSampling(x,y,rate=1/100):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))

    then= datetime.now()
    undersample = NearMiss(sampling_strategy=rate,version=1, n_neighbors=3)
    x_train, y_train = undersample.fit_resample(x, y)
    
    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))
    
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train

def nearMissV2UnderSampling(x,y,rate=1/100):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))

    then= datetime.now()
    undersample = NearMiss(sampling_strategy=rate,version=2, n_neighbors=3)
    x_train, y_train = undersample.fit_resample(x, y)
    
    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))
    
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train

def neighbourhoodCleaningRuleUnderSampling(x,y,rate=1/100):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))

    then= datetime.now()
    undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)
    x_train, y_train = undersample.fit_resample(x, y)
    
    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))

    print(y.value_counts())
    print(y_train.value_counts())
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train

def tomekLinksUnderSampling(x,y):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))

    then= datetime.now()
    undersample = TomekLinks()
    x_train, y_train = undersample.fit_resample(x, y)
    
    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))

    print(y.value_counts())
    print(y_train.value_counts())
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train

def oneSidedSelectionUnderSampling(x,y):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))

    then= datetime.now()
    undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
    x_train, y_train = undersample.fit_resample(x, y)
    
    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))

    print(y.value_counts())
    print(y_train.value_counts())
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train



def randomOverSampling(x,y,rateOverSampling=5):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))

    then= datetime.now()
    oversample = RandomOverSampler(sampling_strategy=fraudRate*rateOverSampling,random_state=42)
    x_train, y_train = oversample.fit_resample(x, y)

    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))

    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train

def randomOverUnderSampling(x,y,rateOverSampling=5,rateUnderSampling=1/100):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))
    
    then= datetime.now()
    over = RandomOverSampler(sampling_strategy=rateOverSampling*fraudRate, random_state=42)
    under = RandomUnderSampler(sampling_strategy=rateUnderSampling, random_state=42) 
    x_over, y_over = over.fit_resample(x, y)
    x_train, y_train = under.fit_resample(x_over, y_over) 

    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))
    
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train

def smoteOverSampling(x,y,rateOverSampling=5):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print("Before Sampling shape {0} fraud rate {1:.5f} ".format(x.shape,fraudRate))

    then= datetime.now()
    oversample = SMOTE(sampling_strategy=fraudRate*rateOverSampling)
    x_train, y_train = oversample.fit_resample(x, y)

    now = datetime.now()
    duration= now - then
    duration_in_s = duration.total_seconds()
    print("Duration {0:.1f} s ".format(duration_in_s))
    
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print("After Sampling shape  {0} fraud rate {1:.5f} ".format(x_train.shape,fraudRate2))
    return x_train, y_train