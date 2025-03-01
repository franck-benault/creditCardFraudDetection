
def amountFiltering(dfTrx,maxAmount):
    dfFiltered=dfTrx[(dfTrx['trx_amount']>=maxAmount)]
    dfRemained=dfTrx[(dfTrx['trx_amount']<maxAmount)]
    return dfRemained

def reversalFiltering(dfTrx):
    dfRemained=dfTrx[(dfTrx['trx_reversal']!='PARTIAL REVERSAL')]
    return dfRemained

def mccFiltering(dfTrx):
    dfRemained=dfTrx[(dfTrx['term_mcc']!=5542)]
    return dfRemained

def fullFiltering(dfTrx,maxAmount=5_000):
    dfTrx=amountFiltering(dfTrx,maxAmount)
    dfTrx=reversalFiltering(dfTrx)
    dfTrx=mccFiltering(dfTrx)
    return dfTrx
    