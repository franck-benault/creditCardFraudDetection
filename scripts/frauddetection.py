import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
from datetime import datetime

## import data 
def read_file(inputFileName):
    dateparse = lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S')
    dfTrx = pd.read_csv(inputFileName, sep=";", parse_dates=['trx_date_time'], date_parser=dateparse)
    
    return dfTrx

