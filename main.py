# -*- coding: utf-8 -*-
#%% imports
import pandas as pd
import os
import src.utils as utils
import src.classifier as clf
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from src.preprocess import TextCleaner
from src.preprocess import ColumnSelector
from src.preprocess import TypeSelector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#%% Read files
path = os.getcwd()
labels = pd.read_table(path + '/resources/data/corpus_machismo_etiquetas_fran.csv', sep=";")
labels = labels[["status_id","categoria"]]
tweets_fields = pd.read_csv(path + '/resources/data/corpus_machismo_frodriguez_atributos_extra.csv', 
                            dtype={'status_id': 'str'})
#%% Cruce de los ficheros
x_cols2 = ['text','source', 'display_text_width', 'respuesta', 'respuesta_screen_name',
          'is_quote', 'is_retweet', 'favorite_count', 'retweet_count', 'hastag_presence',
          'url_presence', 'media_type', 'mentions_presence', 'retweet_favorite_count',
          'retweet_retweet_count', 'retweet_followers_count', 'retweet_verified',
          'protected', 'followers_count', 'friends_count', 'listed_count', 'statuses_count',
          'favourites_count', 'verified', 'categoria']
tweets_fields = utils.change_dtypes(tweets_fields, {'status_id': str})
labels = utils.change_dtypes(labels, {'status_id': str})
tweets_labeled = tweets_fields.merge(labels, on = 'status_id', how = 'inner')
tweets_labeled['respuesta'] = np.where(tweets_labeled['reply_to_status_id'].isnull(), 'no', 'si')
tweets_labeled['respuesta_screen_name'] = np.where(tweets_labeled['reply_to_screen_name'].isnull(), 'no', 'si') 
tweets_labeled['hastag_presence'] = np.where(tweets_labeled['hashtags'].isnull(), 'no', 'si') 
tweets_labeled['url_presence'] = np.where(tweets_labeled['urls_url'].isnull(), 'no', 'si') 
tweets_labeled['mentions_presence'] = np.where(tweets_labeled['mentions_user_id'].isnull(), 'no', 'si') 

#tweets_labeled = tweets_labeled[[]]
#texto_prueba = tweets_labeled.loc[:, ['text', 'categoria']].fillna('MACHISTA')
#texto_prueba = tweets_labeled.loc[1240:1250, x_cols2]
#texto_prueba['text'].str.decode("utf-8")

#texto_prueba['text_processed'] = texto_prueba['text'].apply(
#        lambda row: utils.remove_punctuation(utils.remove_accents(row.decode('utf-8'))))
#%% 
categorical_features = ['source', 'respuesta', 'respuesta_screen_name',
          'is_quote', 'is_retweet', 'hastag_presence', 'url_presence',
          'media_type', 'mentions_presence', 'protected', 'verified']
for f in categorical_features:
    tweets_labeled[f] = tweets_labeled[f].astype("category")

#%% Campos

# Campos:  text, source, display_text_width, reply_to_status_id != null?
# reply_to_screen_name != 0, is_quote (es citado), is_retweet, favorite_count, retweet_count,

#campos que faltan por poner:::::::::
# hastags != 0, 
#urls_url !=0, media_type,
# mentions_user_id != 0,
# , quoted_favorite_count, quoted_retweet_count, quoted_followers_count, quoted_friends_count
#quoted_statuses_count, retweet_source, 
#retweet_favorite_count, retweet_retweet_count
# retweet_followers_count, retweet_friends_count,retweet_statuses_count, retweet_verified
#protected
#followers_count
#friends_count
#listed_count
#statuses_count
#favourites_count
#verified


#%% 
x_cols = ['source', 'display_text_width', 'respuesta', 'respuesta_screen_name',
          'is_quote', 'is_retweet', 'favorite_count', 'retweet_count', 'hastag_presence',
          'url_presence', 'media_type', 'mentions_presence', 'retweet_favorite_count',
          'retweet_retweet_count', 'retweet_followers_count', 'retweet_verified',
          'protected', 'followers_count', 'friends_count', 'listed_count', 'statuses_count',
          'favourites_count', 'verified']

preprocess_pipeline = make_pipeline(
    ColumnSelector(columns=x_cols),
    FeatureUnion(transformer_list=[
        ("numeric_features", make_pipeline(
            TypeSelector(np.number),
            SimpleImputer(strategy="constant"),
            StandardScaler()
        )),
        ("categorical_features", make_pipeline(
            TypeSelector("category"),
            SimpleImputer(strategy="constant", fill_value = "NA"),
            OneHotEncoder(handle_unknown='ignore')
        ))
    ])
)
        
preprocessor = TextCleaner(filter_users=True, filter_hashtags=True, 
                           filter_urls=True, convert_hastags=True, lowercase=True, 
                           replace_exclamation=True, replace_interrogation=True, 
                           remove_accents=True, remove_punctuation=True, replace_emojis=True)   
     
text_pipeline = Pipeline([
    ('column_selection', ColumnSelector('text')),
    ('tfidf', TfidfVectorizer(tokenizer=utils.tokenizer_, 
                                          smooth_idf=True, preprocessor = preprocessor,
                                          norm=None))
])  
    
classifier_pipeline = Pipeline([('feature-union', FeatureUnion([('text-features', text_pipeline), 
                               ('other-features', preprocess_pipeline)
                              ])),
                          ('clf', clf.get_classifier())
                          ])
    
classifier_pipeline.fit(tweets_labeled[x_cols2], tweets_labeled['categoria'])

#predicted = classifier_pipeline.predict(texto_prueba.drop('categoria', axis=1))
#print np.mean(predicted == texto_prueba['categoria']) 

cross_val_score(classifier_pipeline, tweets_labeled[x_cols2], tweets_labeled['categoria'], cv = 10)







