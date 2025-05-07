from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def randomUnderSampling(x,y,rate):
    print("randomUnderSampling before", x.shape)
    undersample = RandomUnderSampler(sampling_strategy=rate,random_state=42)
    x_train, y_train = undersample.fit_resample(x, y)
    print("randomUnderSampling after",x_train.shape)  
    return x_train, y_train

def randomOverSampling(x,y,rate):
    ratio=y.value_counts()[1]/y.value_counts()[0]
    print('fraud rate before ',ratio)

    print(x.shape)  
    oversample = RandomOverSampler(sampling_strategy=rate*ratio,random_state=42)
    x_train, y_train = oversample.fit_resample(x, y)

    print(x_train.shape)  
    ratio2=y_train.value_counts()[1]/y_train.value_counts()[0]
    print('fraud rate after',ratio2)
    return x_train, y_train