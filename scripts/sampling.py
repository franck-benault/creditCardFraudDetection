from imblearn.under_sampling import RandomUnderSampler

def randomUnderSampling(x,y,rate):
    print("randomUnderSampling before", x.shape)
    undersample = RandomUnderSampler(sampling_strategy=rate,random_state=42)
    x_train, y_train = undersample.fit_resample(x, y)
    print("randomUnderSampling after",x_train.shape)  
    return x_train, y_train