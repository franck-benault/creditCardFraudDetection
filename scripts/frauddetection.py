import warnings
warnings.filterwarnings('ignore')
import sourcedata as sd

import pandas as pd 
from datetime import datetime


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


