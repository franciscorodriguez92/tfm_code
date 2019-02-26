# -*- coding: utf-8 -*-
#%% imports
import pandas as pd
import os
import src.preprocess_utils as utils
import src.classifier as clf

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

#%% Read files
path = os.getcwd()
labels = pd.read_table(path + '/resources/data/corpus_machismo_etiquetas.csv', sep=";")
labels = labels[["status_id","categoria"]]
tweets_fields = pd.read_csv(path + '/resources/data/corpus_machismo_frodriguez_atributos_extra.csv', 
                            dtype={'status_id': 'str'})
#%% Cruce de los ficheros

tweets_fields = utils.change_dtypes(tweets_fields, {'status_id': str})
labels = utils.change_dtypes(labels, {'status_id': str})
tweets_labeled = tweets_fields.merge(labels, on = 'status_id', how = 'inner')
#tweets_labeled = tweets_labeled[[]]
texto_prueba = tweets_labeled.loc[:, ['text', 'categoria']].fillna('MACHISTA')
#texto_prueba = tweets_labeled.loc[1240:1250, ['text', 'categoria']].fillna('MACHISTA')
#texto_prueba['text'].str.decode("utf-8")

texto_prueba['text_processed'] = texto_prueba['text'].apply(
        lambda row: utils.remove_punctuation(utils.remove_accents(row.decode('utf-8'))))

#%%

LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=utils.tokenizer_, 
                                          smooth_idf=True, preprocessor = None,
                                          norm=None)),
                ('clf', clf.get_classifier()),
            ])

LogReg_pipeline.fit(texto_prueba['text_processed'],texto_prueba['categoria'])

#clf.cross_validation(LogReg_pipeline, texto_prueba['text_processed'], texto_prueba['categoria'], cv = 5)

cross_val_score(LogReg_pipeline, texto_prueba['text_processed'], texto_prueba['categoria'], cv = 5)

#%% Add tfidf features
v = TfidfVectorizer(tokenizer=utils.tokenizer_, 
                                          smooth_idf=True, preprocessor = None,
                                          norm=None)
x = v.fit_transform(texto_prueba['text_processed'])

df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
#df.drop('text', axis=1, inplace=True)
res = pd.concat([texto_prueba, df1], axis=1).fillna('MACHISTA')


random_pipeline = Pipeline([

                ('clf', clf.get_classifier('random_forest'))
            ])

random_pipeline.fit(res.drop('categoria', axis = 1),res['categoria'])

cross_val_score(random_pipeline, res.drop('categoria', axis = 1), res['categoria'], cv = 5)

#%% Add tfidf features

from src.preprocess import TextCleaner

preprocessor = TextCleaner(filter_users=True, filter_hashtags=True, 
                           filter_urls=True, convert_hastags=True, lowercase=True, 
                           replace_exclamation=True, replace_interrogation=True, 
                           remove_accents=True, remove_punctuation=True, replace_emojis=True)
v = TfidfVectorizer(tokenizer=utils.tokenizer_, 
                                          smooth_idf=True, preprocessor = preprocessor,
                                          norm=None)

x = v.fit_transform(texto_prueba['text_processed'])

df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
#df.drop('text', axis=1, inplace=True)
res = pd.concat([texto_prueba, df1], axis=1).fillna('MACHISTA')


random_pipeline = Pipeline([

                ('clf', clf.get_classifier('random_forest'))
            ])

random_pipeline.fit(res.drop('categoria', axis = 1),res['categoria'])

cross_val_score(random_pipeline, res.drop('categoria', axis = 1), res['categoria'], cv = 5)

#%% Campos

# Campos: status_id, screen_name?, text, source, display_text_width, reply_to_status_id != null?
# reply_to_screen_name != 0, is_quote (es citado), is_retweet, favorite_count, retweet_count,
# hastags != 0, urls_url !=0, media_type, mentions_user_id != 0, mentions_screen_name != 0,
# quoted_source, quoted_favorite_count, quoted_retweet_count, quoted_followers_count, quoted_friends_count
#quoted_statuses_count, retweet_source, retweet_favorite_count, retweet_retweet_count
# retweet_followers_count, retweet_friends_count,retweet_statuses_count, retweet_verified
#protected
#followers_count
#friends_count
#listed_count
#statuses_count
#favourites_count
#account_created_at
#verified
