highRiskMccFolliwingCommerceGate= [
    4411,	
    4511,	
    4582,
    4722,
    4812,	
    4814,	
    4816,	
    4829,	
    5094,
    5122,
    5192,
    5511,
    5712,
    5912,
    5962,
    5964,	
    5966,	
    5967,
    5968,
    5969,
    5921,
    5993,
    6051,
    7021,
    7273,	
    7375,	
    7399,
    7841,
    7922,	
    7991,
    7994,
    7995,
    8099,
    8299,	
    8398,	
    9399
]


def get_highrisk_mcc_group(code):  
    count=highRiskMccFolliwingCommerceGate.count(code)
    if (count > 0):
        return "HIGH_RISK"
    else:
        return "LOW_RISK"

mccATM=6011
def get_mcc_group_ATM(code):
    if (code==mccATM):
        return "ATM"
    else:
        return "OTHER"



def get_mcc_group_citybank(code):
    if(code<=1499):
        return "AGRICULTURAL"  
        
    if (code <=2999):
        return "CONTRACTED_SERVICES"  

    if((code>=3000) & (code<=3299)):
        return 'AIRLINES'

    if((code>=3300) & (code<=3499)):
        return 'CAR_RENTAL'

    if((code>3500) &(code<=3999)):
        return 'LODGING'

    if ((code>=4000) & (code<=4799)):
        return "TRANSPORTATION_SERVICES"  

    if ((code>=4800) & (code<=4999)):
        return "UTILITY_SERVICES" 

    if((code>=5000) & (code<=5599)):
        return 'RETAIL_OUTLET_SERVICES'

    if((code>=5600) & (code<=5699)):
        return 'CLOTHING_STORES'

    if((code<=5700)& (code<=7299)):
        return 'MISCELLANOUS_STORES'

    if((code<=7300) & (code<=7999)):
        return 'BUSINESS_SERVICES'

    if((code<=8000) & (code<=8999)):
        return 'PROFESSIONAL_SERVICES'

    if((code<=9000) & (code<=9999)):
        return 'GOUVERNEMENT_SERVICES'
        
    return 'OTHER'