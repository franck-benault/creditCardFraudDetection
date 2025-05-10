from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def draw2DPCAScaterPlot(x,y):
    pca = PCA(n_components=2)
    sc = StandardScaler()
    x2 = sc.fit_transform(x)
    x_pca = pca.fit_transform(x2)

    # giving a larger plot
    plt.figure(figsize=(8, 6))

    plt.scatter(x_pca[:, 0], x_pca[:, 1],
            c=y,
            cmap='plasma')

    # labeling x and y axes
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

def randomUnderSampling(x,y,rate=1/100):
    print("randomUnderSampling shaoe before", x.shape)
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print('fraud rate before  {0:.5f} '.format(fraudRate))
    undersample = RandomUnderSampler(sampling_strategy=rate,random_state=42)
    x_train, y_train = undersample.fit_resample(x, y)
    print("randomUnderSampling shape after",x_train.shape) 
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print('fraud rate after {0:.5f} '.format(fraudRate2))
    return x_train, y_train

def randomOverSampling(x,y,rate=5):
    ratio=y.value_counts()[1]/y.value_counts()[0]
    print('fraud rate before ',ratio)

    print(x.shape)  
    oversample = RandomOverSampler(sampling_strategy=rate*ratio,random_state=42)
    x_train, y_train = oversample.fit_resample(x, y)

    print(x_train.shape)  
    ratio2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print('fraud rate after',ratio2)
    return x_train, y_train

def randomOverUnderSampling(x,y,rateOverSampling=5,rateUnderSampling=1/100):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print('fraud rate before  {0:.5f} '.format(fraudRate))
    #------------------------------------------------------
    over = RandomOverSampler(sampling_strategy=rateOverSampling*fraudRate, random_state=42)
    under = RandomUnderSampler(sampling_strategy=rateUnderSampling, random_state=42) 
    x_over, y_over = over.fit_resample(x, y)

    x_train, y_train = under.fit_resample(x_over, y_over) 
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print('fraud rate after {0:.5f} '.format(fraudRate2))
    print('shape after',x_train.shape) 
    return x_train, y_train

def smoteOverSampling(x,y,rateOverSampling=5):
    fraudRate=y.value_counts()[1]/y.value_counts()[0]
    print('fraud rate before  {0:.5f} '.format(fraudRate))
    oversample = SMOTE(sampling_strategy=fraudRate*rateOverSampling)
    print('shape before',x.shape)
    print(y.value_counts())
    x_train, y_train = oversample.fit_resample(x, y)
    print('shape after',x_train.shape)
    print(y_train.value_counts())
    fraudRate2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print('fraud rate after {0:.5f} '.format(fraudRate2))
    return x_train, y_train