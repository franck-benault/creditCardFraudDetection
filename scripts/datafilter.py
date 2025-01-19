
def amountFiltering(dfTrx,maxAmount):
    dfFiltered=dfTrx[(dfTrx['trx_amount']>=maxAmount)]
    dfRemained=dfTrx[(dfTrx['trx_amount']<maxAmount)]
    return dfRemained

def fullFiltering(dfTrx,maxAmount=5_000):
    return amountFiltering(dfTrx,maxAmount)
    