europeanCountry = [
'BGR',    
'DNK',   
'SWE', 
'LTU', 
'HUN', 
'LUX', 
'ESP', 
'CYP', 
'DEU', 
'FRA',  
'EST', 
'NLD',  
'IRL',  
'AUT',
'HRV',
'FIN',
'GRC',
'ITA',
'LVA',
'MLT',
'POL',
'CZE',
'SVK',
'SVN',
'PRT',
'ROU'
]

def get_country_group(country_code):  
    count=europeanCountry.count(country_code)
    if (country_code=='BEL'):
        return 'BELGIUM'
    if (count > 0):
        return "EUROPE"
    else:
        return "WORLD"