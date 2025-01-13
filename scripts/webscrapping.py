def extractGoogleScolarFigureNbArticles(input):
    words = input.split("About ")   
    #print(words[1])
    words = words[1].split(" result")  
    #print(words[0])
    number_str = words[0].replace('.', '')
    number_int = int(number_str)
    #print(number_int)  # Output: 10
    return number_int