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
    #dfTrx=dfTrx.drop(columns=['term_country'])
    
    dfTrx['mcc_group'] = dfTrx['term_mcc'].apply(mccmanagement.get_mcc_group_citybank)
    dfTrx['mcc_group'] = np.where(dfTrx.term_mcc.isin([mccmanagement.mccATM]),'ATM',dfTrx['mcc_group'] )
    dfTrx['mcc_group'] = np.where(dfTrx.mcc_group.isin(['AGRICULTURAL']), 'OTHER',dfTrx['mcc_group'] )
    dfTrx['mcc_group'] = np.where(dfTrx.mcc_group.isin(['CONTRACTED_SERVICES']), 'OTHER',dfTrx['mcc_group'] )
    dfTrx['mcc_group'] = np.where(dfTrx.mcc_group.isin(['AIRLINES']),'OTHER',dfTrx['mcc_group'] )
    dfTrx['mcc_group'] = np.where(dfTrx.mcc_group.isin(['CAR_RENTAL']), 'OTHER',dfTrx['mcc_group'] )
    dfTrx['mcc_group'] = np.where(dfTrx.mcc_group.isin(['LODGING']),'OTHER',dfTrx['mcc_group'] )
    dfTrx['mcc_group'] = np.where(dfTrx.mcc_group.isin(['TRANSPORTATION_SERVICES']), 'OTHER',dfTrx['mcc_group'] )
    dfTrx['mcc_group'] = np.where(dfTrx.mcc_group.isin(['MISCELLANOUS_STORES']), 'OTHER',dfTrx['mcc_group'] )
    dfTrx['mcc_group'] = np.where(dfTrx.mcc_group.isin(['BUSINESS_SERVICES']), 'OTHER',dfTrx['mcc_group'] )
    #dfTrx= dfTrx.drop(columns=['term_mcc'])
    return dfTrx

def fixTrx_reversal(trx_reversal):
    if(trx_reversal=='FULL REVERSAL'):
        return 'FULL_REVERSAL'
    if(trx_reversal=='NO REVERSAL'):
        return 'NO_RESERSAL'
    else:
        return 'PARTIAL_REVERSAL'

def reversal_fix(dfTrx):
    dfTrx['trx_reversal'] = dfTrx['trx_reversal'].apply(fixTrx_reversal)
    return dfTrx
    
def category_encoding(dfTrx):
    dfTrx=pd.get_dummies(dfTrx,columns=['card_brand','country_group','mcc_group','trx_reversal','clusterCardHolder','clusterMerchant'], dtype = int)
    return dfTrx

def amount_transformation(dfTrx):
    dfTrx['trx_amount_log10']=np.log10(1+dfTrx['trx_amount'])
    dfTrx= dfTrx.drop(columns=['trx_amount'])
    return dfTrx

def ecom(dfTrx):
    dfTrx['ecom'] =np.where(dfTrx.card_entry_mode.isin([5,6,9]), 1,0)
    dfTrx= dfTrx.drop(columns=['card_entry_mode'])
    return dfTrx

def remove_column_not_yet_managed(dfTrx):
    dfTrx= dfTrx.drop(columns=['TRX_3D_SECURED','trx_accepted','trx_cnp','trx_response_code',
                             'ecom_indicator','trx_authentication','pos_entry_mode','ch_present',
                             'acceptor_id'])
    return dfTrx

def join_card_holder_profile(dfTrx,cardHolderProfileFileName):
    dfCardProfile = pd.read_csv('../data/processed/'+cardHolderProfileFileName)
    dfTrx=pd.merge(dfTrx, dfCardProfile, left_on='card_pan_id', right_on='card_pan_id', how='left')
    dfTrx['clusterCardHolder'] = dfTrx['clusterCardHolder'].apply(lambda x: 'UNKNOWN' if pd.isnull(x) == True else x)

    return dfTrx


def join_merchant_profile(dfTrx,merchantProfileFileName):
    dfMerchant = pd.read_csv('../data/processed/'+merchantProfileFileName)
    dfTrx=pd.merge(dfTrx, dfMerchant, left_on=['acceptor_id','term_mcc','term_country'],
                right_on=['acceptor_id','term_mcc','term_country'], how='left')
    dfTrx['clusterMerchant'] = dfTrx['clusterMerchant'].apply(lambda x: 'UNKNOWN' if pd.isnull(x) == True else x)

    return dfTrx

def previous_trx(dfTrx):
    sorted_df = dfTrx.sort_values(by=['card_pan_id','trx_date_time'])
    dfTrx['previous_trx']=0
    for i in range(1,2,1):
        sorted_df['card_pan_id1'] = sorted_df['card_pan_id'].shift(-i)
        sorted_dfTemp=sorted_df[(sorted_df['card_pan_id']==sorted_df['card_pan_id1'])]
        dfTrx['previous_trx']=np.where(sorted_df.index.isin(sorted_dfTemp.index),i,dfTrx['previous_trx'])
    return dfTrx


def full_import_and_clean(inputFileName,cardHolderProfileFileName, merchantProfileFileName):
    dfTrx = read_file(inputFileName)
    dfTrx = remove_columns(dfTrx)
    dfTrx = fill_missing_values(dfTrx)
    dfTrx = category_grouping(dfTrx)
    dfTrx = reversal_fix(dfTrx)
    dfTrx = ecom(dfTrx)
    dfTrx = join_card_holder_profile(dfTrx,cardHolderProfileFileName)
    dfTrx = join_merchant_profile(dfTrx,merchantProfileFileName)
    dfTrx = category_encoding(dfTrx)
    dfTrx = amount_transformation(dfTrx)
    dfTrx = previous_trx(dfTrx)
    dfTrx = remove_column_not_yet_managed(dfTrx)
    
    return dfTrx