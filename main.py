import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import plotly.express as px
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
import re
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils

from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

# Files we will use!
train_path = 'train.ft.txt.bz2'
test_path = 'test.ft.txt'

# Read Data
train_data = pd.read_csv(train_path, compression='bz2', delimiter='\t') # delimiter marks the beginning/end of smth
test_data = pd.read_csv(test_path, delimiter='\t')
# print(train_data.head())

# print("Train data shape" ,train_data.shape)
# print("Test data shape" ,test_data.shape)

# This function is to convert the data into 2 columns: label (based on star rating range; 1 OR 2) & text (user review)
# Takes file OR train/test data, and loops over text in file to split texts and labels.

# Data Preparing
def process_data(file) :
    data = []
    #during the entire file duration
    for index, row in file.iterrows():
        line = row[0] # raw data we don't need

        # the text rating is followed by space
        label, text = line.split(' ', 1)

        #remove __label__ and just keep number
        label = label.replace('__label__', '')

        #add the two
        data.append((label, text.strip()))
    
    #naming for the table
    cols = ['label', 'review']
    return pd.DataFrame(data, columns=cols)

# Data Cleaning
def text_cleaning(text):
    
    #apply lowercase
    text = text.lower()  
    
    #remove special characters and numbers and extra whitespace
    pattern_punc = r'[^A-Za-z\s]'
    text = re.sub(pattern_punc, '', text).strip()
    return text

# Show data in a table format (better readability) => Using process_data
train = process_data(train_data)
# print(train.head())

test = process_data(test_data)
# print(test.head())

#Using text_cleaning
train['review_cleaned_ver'] = train['review'].apply(text_cleaning)
# print(train.head())

train['review_cleaned_ver'] = test['review'].apply(text_cleaning)
# print(test.head())

