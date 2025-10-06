#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier, XGBRegressor
import unidecode

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
from unidecode import unidecode
text_columns =  ['country', 'description','province','region_1','variety','winery']


def remove_accents(text):
    if isinstance(text, str):
        return unidecode(text)
    return text


for col in text_columns:
    df[col]= df[col].apply(remove_accents)

#%%
print(df.variety.value_counts())

#%%

def correct_grape_names(row):
    regexp = [
        r'shiraz', r'ugni blanc', r'cinsaut', r'carinyena',
        r'^ribolla$', r'palomino', r'turbiana', r'verdelho', r'viura',
        r'pinot bianco|weissburgunder',
        r'garganega|grecanico',
        r'moscatel', r'moscato',
        r'melon de bourgogne',
        r'trajadura|trincadeira',
        r'cannonau|garnacha',
        r'grauburgunder|pinot grigio',
        r'pinot noir|pinot nero',
        r'colorino',
        r'mataro|monastrell',
        r'mourv(\w+)'
    ]

    grapename = [
        'syrah', 'trebbiano', 'cinsault', 'carignan',
        'ribolla gialla', 'palomino', 'verdicchio', 'verdejo', 'macabeo',
        'pinot blanc',
        'garganega',
        'muscatel', 'muscat',
        'muscadet',
        'treixadura',
        'grenache',
        'pinot gris',
        'pinot noir',
        'lambrusco',
        'mourvedre', 'mourvedre'
    ]

    f = row
    for exsearch, gname in zip(regexp, grapename):
        f = re.sub(exsearch, gname, f)
    return f

#%%
name_pairs = [
    ('spatburgunder', 'pinot noir'),
    ('garnacha', 'grenache'),
    ('pinot nero', 'pinot noir'),
    ('alvarinho', 'albarino'),
    ('assyrtico', 'assyrtiko'),
    ('black muscat', 'muscat hamburg'),
    ('kekfrankos', 'blaufrankisch'),
    ('garnacha blanca', 'grenache blanc'),
    ('garnacha tintorera', 'alicante bouschet'),
    ('sangiovese grosso', 'sangiovese')
]

def correct_by_pairs(text):
    if isinstance(text, str):
        for old, new in name_pairs:
            text = re.sub(rf"\b{old}\b", new, text, flags=re.IGNORECASE)
    return text

df['variety'] = df['variety'].apply(correct_by_pairs)

#%%
print(len(df))
#%%
print(len(df.variety.value_counts()))
#%%
df=df.groupby('variety').filter(lambda x: len(x) > 150)
print(df.shape)

#%%
graphs= df['variety'].unique().tolist()
print(len(graphs))

#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.barplot(
    x=df['variety'].value_counts().index,
    y=df['variety'].value_counts().values
)
plt.xticks(rotation=45)
plt.title("Top 15 Grape Varieties after Filtering (>150 samples)")
plt.show()

#%%
df.variety.value_counts()
#%%
colour_map = {
    'aglianico': 'red', 'albarino': 'white', 'barbera': 'red', 'cabernet franc': 'red',
    'cabernet sauvignon': 'red', 'carmenere': 'red', 'chardonnay': 'white', 'chenin blanc': 'white',
    'corvina, rondinella, molinara': 'red', 'gamay': 'red', 'garganega': 'white',
    'gewurztraminer': 'white', 'glera': 'white', 'grenache': 'red', 'gruner veltliner': 'white',
    'malbec': 'red', 'merlot': 'red', 'mourvedre': 'red', 'muscat': 'white', 'nebbiolo': 'red',
    "nero d'avola": 'red', 'petite sirah': 'red', 'pinot blanc': 'white', 'pinot gris': 'white',
    'pinot grigio': 'white', 'pinot noir': 'red', 'port': 'red', 'prosecco': 'white',
    'riesling': 'white', 'sangiovese': 'red', 'sauvignon blanc': 'white', 'syrah': 'red',
    'tempranillo': 'red', 'torrontes': 'white', 'verdejo': 'white', 'viognier': 'white',
    'zinfandel': 'red',
    'rose': 'rose',
    'bordeaux-style red blend': 'red',
    'bordeaux-style white blend': 'white',
    'rhone-style red blend': 'red',
    'rhone-style white blend': 'white',
    'tempranillo blend': 'red',
    'cabernet blend': 'red',
    'cabernet sauvignon-merlot': 'red',
    'sparkling blend': 'white',
    'champagne blend': 'white',
    'meritage': 'red',
    'primitivo': 'red',
    'dolcetto': 'red',
    'sauvignon': 'white',

    # extra unmapped from your warning
    'red blend': 'red',
    'white blend': 'white',
    'shiraz': 'red',
    'moscato': 'white',
    'portuguese red': 'red',
    'portuguese white': 'white',
}

df['colour'] = df['variety'].map(colour_map)

unmapped = set(df['variety']) - set(colour_map.keys())
if unmapped:
    print("⚠️ Still unmapped:", unmapped)

df = df.drop(columns=[c for c in ['red', 'white', 'rose'] if c in df.columns])

colour_dummies = pd.get_dummies(df['colour']).astype(int)

df = pd.concat([df, colour_dummies], axis=1)

print(df[['variety', 'colour', 'red', 'white', 'rose']].head(10))

print("\nColour distribution:")
print(df['colour'].value_counts())

#%%

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
        return " ".join(filtered)
    return text

df['description_clean'] = df['description'].apply(remove_stopwords)

print(df[['description', 'description_clean']].head(5))
#%%
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        filtered = [
            lemmatizer.lemmatize(w.lower())
            for w in tokens if w.lower() not in stop_words and w.isalpha()
        ]
        return " ".join(filtered)
    return text

df['description_text']= df['description_clean'].apply(clean_text)

print(df[['description','description_clean']].head(5))

#%%

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words('english'))
extra_stopwords = ['wine', 'fruit', 'flavor', 'aromas', 'palate']
stop_words = stop_words.union(extra_stopwords)

defTags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('R'):
        return 'r'  # adverb
    return 'n'  # default noun



class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):

        tokens = word_tokenize(doc.lower())


        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]


        tagged = nltk.pos_tag(tokens)


        tagged = [(word, tag) for word, tag in tagged if tag in defTags]


        lemmas = [self.wnl.lemmatize(word, penn_to_wn(tag)) for word, tag in tagged]

        return lemmas
#%%
# shape of final dataset
print("Shape:", df.shape)

# first few rows
print("\nHead of df:")
print(df.head(5))

# check columns
print("\nColumns:", df.columns.tolist())

# check variety distribution
print("\nTop 10 varieties:")
print(df['variety'].value_counts().head(10))

# check colour distribution
print("\nColour distribution:")
print(df['colour'].value_counts())

# check description after cleaning
print("\nOriginal vs Cleaned description:")
print(df[['description', 'description_clean']].head(5))

#%%


#%%


#%%
# classification models - model buidling


# Target variable
y = df['variety']

# Text column for vectorization
text_data = df['description_clean']

# Numeric columns
num_features = df[['price', 'red', 'white', 'rose']]

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF with LemmaTokenizer (we already defined this above)
tfidf = TfidfVectorizer(
    tokenizer=LemmaTokenizer(),
    ngram_range=(1, 3),   # unigrams, bigrams, trigrams
    min_df=5,             # ignore very rare words
    max_df=0.8,           # ignore extremely common words
    sublinear_tf=True,
    norm='l2'
)

# Fit-transform on description text
X_text = tfidf.fit_transform(text_data)
print("TF-IDF matrix shape:", X_text.shape)
#%%
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Scale numeric features
scaler = StandardScaler(with_mean=False)
X_num_scaled = scaler.fit_transform(num_features)

# Combine text + numeric features
X = hstack([X_text, X_num_scaled])
print("Final feature matrix shape:", X.shape)

#%%

from sklearn.preprocessing import LabelEncoder

# Initialize encoder
label_encoder = LabelEncoder()

# Fit and transform target labels
y_encoded = label_encoder.fit_transform(y)

print("Example mapping:")
print(dict(zip(label_encoder.classes_[:10], range(10))))
print("Encoded y shape:", y_encoded.shape)

#%%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


#%%

print("Train:", X_train.shape, "Test:", X_test.shape)
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

#%%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

#%%
from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(df['variety'].unique()),
    eval_metric='mlogloss',
    learning_rate=0.1,
    max_depth=8,
    n_estimators=250,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))


#%%
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf,
    "XGBoost": xgb
}

from sklearn.metrics import accuracy_score

for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")
#%%
from sklearn.pipeline import Pipeline

xgb_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        tokenizer=LemmaTokenizer(),
        ngram_range=(1,3),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True,
        norm='l2'
    )),
    ('clf', XGBClassifier(
        objective='multi:softmax',
        num_class=len(df['variety'].unique()),
        eval_metric='mlogloss',
        random_state=42
    ))
])

#%%
param_grid = {
    'clf__learning_rate': [0.05, 0.1, 0.2],
    'clf__max_depth': [6, 8, 10],
    'clf__n_estimators': [150, 250, 350],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_pipe,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(df['description_clean'], df['variety'])

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
#%%
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_xgb, labels=y.unique())
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="YlGnBu")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


