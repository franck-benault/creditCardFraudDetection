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
    dfTrx['term_mcc'] = dfTrx['term_mcc'].astype('str')

    dfTrx['ecom_indicator_group'] = np.where(dfTrx.ecom_indicator.isin([2,61]), 1,0)
    dfTrx=dfTrx.drop(columns=['ecom_indicator'])
    
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

def trx_authentication_group(dfTrx):
    dfTrx['trx_authentication_group'] = np.where(dfTrx.trx_authentication.isin(['S']), 1,0)
    dfTrx= dfTrx.drop(columns=['trx_authentication'])
    return dfTrx

def ch_present_group(dfTrx):
    dfTrx['ch_present_group'] = np.where(dfTrx.ch_present.isin([2, 3, 1, 4]), 1,0)
    dfTrx= dfTrx.drop(columns=['ch_present'])
    return dfTrx

def trx_response_code_group(dfTrx):
    dfTrx['trx_response_code_group'] = np.where(dfTrx.trx_response_code.isin(['13', '83', '59', '63', 'N7']), 1,0)
    dfTrx= dfTrx.drop(columns=['trx_response_code'])
    return dfTrx

def pos_entry_mode_group(dfTrx):
    dfTrx['pos_entry_mode_group'] = np.where(dfTrx.pos_entry_mode.isin([801, 11, 12, 10, 810, 812, 100]), 1,0)
    dfTrx= dfTrx.drop(columns=['pos_entry_mode'])
    return dfTrx

def trx_winning_bit_group(dfTrx):
    dfTrx['trx_winning_bit_group'] = np.where(dfTrx.trx_winning_bit.isin([138, 48, 140, 129, 178]), 1,0)
    dfTrx= dfTrx.drop(columns=['trx_winning_bit'])
    return dfTrx

def remove_columns(dfTrx):

    # remove db_uuid? acceptor_id used for merchant group link
    dfTrx= dfTrx.drop(columns=['acceptor_id'])

    # too low IV ( previous_trx)
    dfTrx= dfTrx.drop(columns=['previous_trx','magstripe_fallback'])


    #dfTrx= dfTrx.drop(columns=['trx_winning_bit'])

    # Remove columns with corelation to another or grouping columns
    dfTrx= dfTrx.drop(columns=['card_brand_VIS'])
    print(dfTrx.shape)
    
    return dfTrx

def join_card_holder_profile(dfTrx,cardHolderProfileFileName):
    dfCardProfile = pd.read_csv('../data/processed/'+cardHolderProfileFileName)
    dfTrx=pd.merge(dfTrx, dfCardProfile, left_on='card_pan_id', right_on='card_pan_id', how='left')
    dfTrx['clusterCardHolder'] = dfTrx['clusterCardHolder'].apply(lambda x: 'UNKNOWN' if pd.isnull(x) == True else x)

    return dfTrx


def join_merchant_profile(dfTrx,merchantProfileFileName):
    dfMerchant = pd.read_csv('../data/processed/'+merchantProfileFileName)
    dfMerchant['term_mcc'] = dfMerchant['term_mcc'].astype('str')
    dfTrx=pd.merge(dfTrx, dfMerchant, left_on=['acceptor_id','term_mcc','term_country'],
                right_on=['acceptor_id','term_mcc','term_country'], how='left')
    dfTrx['clusterMerchant'] = dfTrx['clusterMerchant'].apply(lambda x: 'UNKNOWN' if pd.isnull(x) == True else x)

    return dfTrx



def word_not_present(wordT,model):
    if (wordT in model.wv.key_to_index):
        return wordT
    else:
        #print(wordT)
        return 'BEL5411'

def calculDistance(input0,input1,input2,input3,model):
    if(input0!=input1):
        return 1.1
    else:
        return model.wv.similarity(input2,input3)

def aggregate_cardholder(dfTrx):
    max=18
    sorted_df = dfTrx.sort_values(by=['card_pan_id','trx_date_time'])


    for i in np.arange(1,max+1,1):
        sorted_df['card_pan_id'+str(i)] = sorted_df['card_pan_id'].shift(-1*i)
        sorted_df['trx_amount'+str(i)] = sorted_df['trx_amount'].shift(-1*i)

    dfTrx['previousTrxAmountSum']=0
    dfTrx['nbPreviousTrx']=0

    for i in np.arange(1,max+1,1):
        sorted_dfTemp1=sorted_df[(sorted_df['card_pan_id']==sorted_df['card_pan_id'+str(i)])]
        dfTrx['nbPreviousTrx']=np.where(sorted_df.index.isin(sorted_dfTemp1.index),i,dfTrx['nbPreviousTrx'])
        dfTrx['previousTrxAmountSum']=np.where(sorted_df.index.isin(sorted_dfTemp1.index),
            sorted_df['trx_amount'+str(i)]+dfTrx['previousTrxAmountSum'],dfTrx['previousTrxAmountSum'])

    dfTrx['previousTrxAmountSumLog']=np.log10(1+dfTrx['previousTrxAmountSum'])
    dfTrx['previousTrxAmountSumLog'].fillna(0.0, inplace=True)
    dfTrx= dfTrx.drop(columns=['previousTrxAmountSum'])
    return dfTrx

def previous_trx(dfTrx):
    dfTrx['previous_trx']=0
    dfTrx['wordV2']=dfTrx['term_country']+dfTrx['term_mcc']
    sorted_df = dfTrx.sort_values(by=['card_pan_id','trx_date_time'])

    sorted_df['card_pan_id1'] = sorted_df['card_pan_id'].shift(-1)
    sorted_df['wordV2P'] = sorted_df['wordV2'].shift(-1)
    dfTrx['wordV2P'] = sorted_df['wordV2P']
    dfTrx['card_pan_id1']=sorted_df['card_pan_id1']

    sorted_dfTemp=sorted_df[(sorted_df['card_pan_id']==sorted_df['card_pan_id1'])]
    dfTrx['previous_trx']=np.where(sorted_df.index.isin(sorted_dfTemp.index),1,dfTrx['previous_trx'])

    dfTrx['distancePrevTrx']=1.1
    import pickle
    model = pickle.load(open('../data/processed/wordV2Model', 'rb'))

    dfTrx['wordV2'] = dfTrx['wordV2'].apply(lambda x:word_not_present(x,model))
    dfTrx['wordV2P'] = dfTrx['wordV2P'].apply(lambda x:word_not_present(x,model))

    dfTrx['distancePrevTrx']= dfTrx.apply(lambda x: calculDistance(x.card_pan_id,x.card_pan_id1,x.wordV2P,x.wordV2,model), axis=1)
    dfTrx= dfTrx.drop(columns=['wordV2','wordV2P','card_pan_id1'])
    return dfTrx


def full_import_and_clean(inputFileName,cardHolderProfileFileName, merchantProfileFileName):
    dfTrx = read_file(inputFileName)
    dfTrx = fill_missing_values(dfTrx)
    dfTrx = category_grouping(dfTrx)
    dfTrx = reversal_fix(dfTrx)
    dfTrx = ecom(dfTrx)
    dfTrx = trx_authentication_group(dfTrx) 
    dfTrx = ch_present_group(dfTrx)
    dfTrx = trx_response_code_group(dfTrx)
    dfTrx = pos_entry_mode_group(dfTrx)
    dfTrx = trx_winning_bit_group(dfTrx)
    dfTrx = join_card_holder_profile(dfTrx,cardHolderProfileFileName)
    dfTrx = join_merchant_profile(dfTrx,merchantProfileFileName)
    dfTrx = category_encoding(dfTrx)
    dfTrx = aggregate_cardholder(dfTrx)
    dfTrx = amount_transformation(dfTrx)
    dfTrx = previous_trx(dfTrx)
    dfTrx = remove_columns(dfTrx) 
    return dfTrx