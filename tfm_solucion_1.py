# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:46:47 2018

@author: Francisco Miguel Rodríguez Sánchez
"""

#%% Imports

from nltk.tokenize import TweetTokenizer
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import unidecode
from nltk import PorterStemmer
import string
import src.preprocess_utils as utils


#%% Clean text

def clean_text(text):
    return text.lower().replace('\n',' ').translate(None, string.punctuation)


def tokenizador_(text):
    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(
            unidecode.unidecode(unidecode.unidecode(text)))
    tokens_clean = utils.replace_abb(tokens)
    return [PorterStemmer().stem(word) for word in tokens_clean]


#%% Read
path = os.getcwd()
df = pd.read_csv(path+'/resources/data/MW_corpus_machismo_frodriguez_2018-10-26.csv')
df_sample = df.sample(100).reset_index()
df_sample['clase'] = ['machismo', 'neutro'] * 50

#%% tf-idf
df_sample_text = df_sample[['text','clase']]
df_sample_text['text_lower'] = df_sample['text'].apply(lambda row: clean_text(row))
#df_sample_text['tokens'] = df_sample_text['text_lower'].apply(
#        lambda row: TweetTokenizer().tokenize(row))
#%% tf-idf
stop_words = set(stopwords.words('Spanish'))

Train_Textlist, Test_Textlist = train_test_split(df_sample_text,
test_size= 0.3, random_state=13)

tfidf = TfidfVectorizer(tokenizer=tokenizador_, smooth_idf=True,
norm=None, stop_words=stop_words)

Train_corpus = Train_Textlist['text_lower'].values.tolist()
Test_corpus = Test_Textlist['text_lower'].values.tolist()

Train_tfidf_X = tfidf.fit_transform(Train_corpus)
Test_tfidf_X = tfidf.transform(Test_corpus)

Train_tfidf = pd.DataFrame(Train_tfidf_X.A, columns=tfidf.get_feature_names())
Train_tfidf.index = Train_Textlist.index
#Train_tfidf['CATEGORIA'] = Train_Textlist['CATEGORIA']
Test_tfidf = pd.DataFrame(Test_tfidf_X.A, columns=tfidf.get_feature_names())
Test_tfidf.index = Test_Textlist.index
#Test_tfidf['CATEGORIA'] = Test_Textlist['CATEGORIA']

#%% Clasificador regresión logística
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import classification_report

lr = LogisticRegression()
lr.fit(Train_tfidf, Train_Textlist['clase'])
#Predicción de los valores de testeo:
y_pred = lr.predict(Test_tfidf)
#- Precisión f1 del modelo:
f1_Score = f1_score(y_pred, Test_Textlist['clase'],
labels=Test_Textlist['clase'].unique(),
average='micro')
print 'Precision f1', f1_Score
print('accuracy %s' % accuracy_score(y_pred, Test_Textlist['clase']))
print(classification_report(Test_Textlist['clase'], 
                        y_pred,target_names=Test_Textlist['clase'].unique()))

#%% Clasificador NB
#from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
#import seaborn as sns
#from sklearn.metrics import classification_report

#Declaración del modelo:
nb = MultinomialNB()
#Entrenamiento:
nb.fit(Train_tfidf, Train_Textlist['clase'])
#Predicción de los valores de testeo:
y_pred = nb.predict(Test_tfidf)
#- Precisión f1 del modelo:
f1_Score = f1_score(y_pred, Test_Textlist['clase'],
labels=Test_Textlist['clase'].unique(),
average='micro')
print 'Precision f1', f1_Score
print('accuracy %s' % accuracy_score(y_pred, Test_Textlist['clase']))
print(classification_report(Test_Textlist['clase'], 
                        y_pred,target_names=Test_Textlist['clase'].unique()))
#%% Clasificador Decision Tree
#from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
#import seaborn as sns
#from sklearn.metrics import classification_report

#Declaración del modelo:
dt = DecisionTreeClassifier()
#Entrenamiento:
dt.fit(Train_tfidf, Train_Textlist['clase'])
#Predicción de los valores de testeo:
y_pred = dt.predict(Test_tfidf)
#- Precisión f1 del modelo:
f1_Score = f1_score(y_pred, Test_Textlist['clase'],
labels=Test_Textlist['clase'].unique(),
average='micro')
print 'Precision f1', f1_Score
print('accuracy %s' % accuracy_score(y_pred, Test_Textlist['clase']))
print(classification_report(Test_Textlist['clase'], 
                        y_pred,target_names=Test_Textlist['clase'].unique()))

#Matriz de Confusión:
# Transform to df for easier plotting
#cm_df = pd.DataFrame(confusion_matrix(Test_tfidf['CATEGORIA'], y_pred),
#index = sorted(Test_tfidf.CATEGORIA.unique()),
#columns = sorted(Test_tfidf.CATEGORIA.unique()))
#plt.figure(figsize=(5.5,4))
#sns.heatmap(cm_df, annot=True)
#plt.title('DecisionTreeClassifier \nAccuracy:{0:.3f}'.
#format(accuracy_score(Test_tfidf['CATEGORIA'],
#y_pred)))
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()















