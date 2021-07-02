import pandas as pd
import numpy as np

import unicodedata
import re
import json
import time
import os
import requests

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords




def wrangle_commit():
    '''
    Wrangles train.csv data into lemmetized 
    
    
    '''
    df =  pd.read_csv('train.csv')
    df = df.drop(columns=['url_legal','license'])
    df['lemmed_text'] = [util.clean_lem_stop(string) for string in df.excerpt]
    
    return df