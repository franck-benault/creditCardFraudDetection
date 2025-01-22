import warnings
warnings.filterwarnings('ignore')
import sourcedata as sd
import countrymanagement as cm
import mccmanagement

import pandas as pd 
import numpy as np
from datetime import datetime


## import data 
def read_file(inputFileName):
    if(sd.source=='KAGGLE'):
        dfTrx = pd.read_csv(inputFileName, sep=",")
    else:
        dateparse = lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S')
        dfTrx = pd.read_csv(inputFileName, sep=";", parse_dates=['trx_date_time'], date_parser=dateparse)
    
    return dfTrx


def remove_columns(dfTrx):
    dfTrx=dfTrx.drop(columns=['issuer_id', 'cluster_profile'])
    return dfTrx

def fill_missing_values(dfTrx):
    dfTrx['mcd_fraud_score'].fillna(np.mean(dfTrx['mcd_fraud_score']), inplace=True)
    dfTrx['vaa_score'].fillna(np.mean(dfTrx['vaa_score']), inplace=True)
    return dfTrx

def category_grouping(dfTrx):
    dfTrx['country_group'] = dfTrx['term_country'].apply(cm.get_country_group)
    dfTrx=dfTrx.drop(columns=['term_country'])
    dfTrx['mcc_group'] = dfTrx['term_mcc'].apply(mccmanagement.get_mcc_group_ATM)
    dfTrx=dfTrx.drop(columns=['term_mcc'])
    return dfTrx
    
def category_encoding(dfTrx):
    dfTrx=pd.get_dummies(dfTrx,columns=['card_brand','country_group','mcc_group'], dtype = int)
    return dfTrx

def remove_column_not_yet_managed(dfTrx):
    dfTrx= dfTrx.drop(columns=['TRX_3D_SECURED','trx_accepted','trx_cnp','trx_response_code','trx_reversal',
                             'ecom_indicator','trx_authentication','pos_entry_mode','card_entry_mode','ch_present',
                            'card_pan_id'])
    return dfTrx


def full_import_and_clean(inputFileName):
    dfTrx = read_file(inputFileName)
    dfTrx = remove_columns(dfTrx)
    dfTrx = fill_missing_values(dfTrx)
    dfTrx = category_grouping(dfTrx)
    dfTrx = category_encoding(dfTrx)
    dfTrx = remove_column_not_yet_managed(dfTrx)
    return dfTrx