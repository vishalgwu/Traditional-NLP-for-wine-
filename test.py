#%%
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import re
import seaborn as sns
import sklearn.metrics as metrics
import sklearn.preprocessing as pre
from fontTools.misc.cython import returns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from string import punctuation
import unicodedata
#%%
data1= pd.read_csv('winemag-data_first150k.csv')
print(data1.head(5) )

"""
data2=pd.read_csv("winemag-data-130k-v2.csv")
print(data2.head(10))

import json
with open('winemag-data-130k-v2.json','r') as json_file:
    data3 = json.load(json_file)
    data3=pd.DataFrame(data3)
print(data3.head(10))
"""

df= data1.drop([ 'Unnamed: 0' , 'designation','points','region_2'], axis=1)
print(df.head(5))

#%%

print(df.shape)
#%%

df=df.drop_duplicates('description')
print(df.shape)

#%%
print(df.columns)
#%%
df=df.dropna( subset= ['description','price','variety'])
#%%
print(df.shape)

#%%
print(df.head(10))

#%%

print(len(df))
#%%
print(df.describe(include='all'))
#%%
print(df.columns)
#%%

for col in ['country', 'description','province', 'region_1', 'variety',
       'winery']:
    df[col]= df[col].str.lower()

print(df.head(5))

#%%




