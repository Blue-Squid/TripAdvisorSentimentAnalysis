"""
Steps: 
    Stop words removed from review text. 
    Reviews will be stemmed and lematized. 
    Result is saved in data as 'sl_data.csv' 
"""

# text processing 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer 

nltk.download('stopwords') 
STEMMER = SnowballStemmer('english') 
STOPWORDS = stopwords.words('english') 

# data analysis   
import pandas as pd 

def find_special_char(text): 
    # helper function - checks special characters in text 
    special_char = "" 
    for i in range(len(text)): 
        if(text[i].isalpha()): 
            continue
        elif(text[i].isdigit()): 
            special_char += text[i] 
        else: 
            special_char += text[i] 
    return set(special_char) 


def clean_text(text, stem=True): 
    """ helper function - 
        removes stopwords, special characters from text 
        replaces words with stems 
    """ 
    CHARS_TO_REMOVE = find_special_char(text) 
    CHARS_TO_REMOVE.remove(" ") 
    TOKENS = list()
    rx = '[' + re.escape(''.join(CHARS_TO_REMOVE)) + ']'
    text = re.sub(rx, "", text) 

    for token in text.split(): 
        if(token not in STOPWORDS): 
            if(stem): 
                TOKENS.append(STEMMER.stem(token)) 
            else: 
                TOKENS.append(token)
    return " ".join(TOKENS) 

if __name__ == '__main__': 
    file = "data/tripadvisor_hotel_reviews.csv" 
    data = pd.read_csv(file) 
    print("Processing text...") 
    data.Review = data.Review.apply(clean_text) 
    data.to_csv("data/cleaned_data.csv", index=False) 
    print("Cleaned data saved to Data folder") 