# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:32:46 2019

@author: Administrador
"""
#%% imports
import pandas as pd
import os
import src.utils as utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.preprocess import TypeSelector

#%% Read files
path = os.getcwd()
labels = pd.read_table(path + '/resources/data/corpus_machismo_etiquetas.csv', sep=";")
labels = labels[["status_id","categoria"]]
tweets_fields = pd.read_csv(path + '/resources/data/corpus_machismo_frodriguez_atributos_extra.csv', 
                            dtype={'status_id': 'str'})
#%% Cruce de los ficheros
x_cols2 = ['text','source', 'display_text_width', 'respuesta', 'respuesta_screen_name',
          'favorite_count', 'retweet_count', 'hastag_presence',
          'url_presence', 'media_type', 'mentions_presence',
          'followers_count', 'friends_count', 'listed_count', 'statuses_count',
          'favourites_count', 'verified', 'categoria']
tweets_fields = utils.change_dtypes(tweets_fields, {'status_id': str})
labels = utils.change_dtypes(labels, {'status_id': str})
tweets_labeled = tweets_fields.merge(labels, on = 'status_id', how = 'inner')
tweets_labeled['respuesta'] = np.where(tweets_labeled['reply_to_status_id'].isnull(), 'no', 'si')
tweets_labeled['respuesta_screen_name'] = np.where(tweets_labeled['reply_to_screen_name'].isnull(), 'no', 'si') 
tweets_labeled['hastag_presence'] = np.where(tweets_labeled['hashtags'].isnull(), 'no', 'si') 
tweets_labeled['url_presence'] = np.where(tweets_labeled['urls_url'].isnull(), 'no', 'si') 
tweets_labeled['mentions_presence'] = np.where(tweets_labeled['mentions_user_id'].isnull(), 'no', 'si') 
tweets_labeled = tweets_labeled[x_cols2]

#%% 
tweets_labeled.describe()

#%% Distribucion de la clase
sns.set(style="darkgrid")
ax = sns.countplot(x="categoria", data=tweets_labeled, palette="BuGn_r")
#%% Tabla de frecuencia de la clase
((pd.value_counts(tweets_labeled["categoria"])/len(tweets_labeled["categoria"]))*100).to_frame().reset_index()

#%% Histograma variables numericas
selector = TypeSelector(np.number)
tweets_labeled_number = selector.fit_transform(tweets_labeled)
tweets_labeled_number["categoria"] = tweets_labeled["categoria"].values
tweets_labeled_number.hist(bins=30, figsize=(15, 6), layout=(2, 4));

#%% Pairplot para variables numericas
sns.pairplot(tweets_labeled_number.fillna(0), hue = "categoria")
#%% Pairplot para dos variables que pueden ser relavantes: "display_text_width", "retweet_count"
sns.pairplot(tweets_labeled_number, hue = "categoria", vars = ["display_text_width", "retweet_count"])
#%% Distribucion clase para variables categoricas
categorical_features = ['respuesta', 'respuesta_screen_name',
          'hastag_presence', 'url_presence',
          'media_type', 'mentions_presence', 'verified', 'source']
tweets_labeled['source'] = tweets_labeled['source'].str.decode('utf-8').astype("category")

for f in categorical_features:
    tweets_labeled[f] = tweets_labeled[f].astype("category")
tweets_labeled_categorical = tweets_labeled[categorical_features]
tweets_labeled_categorical["categoria"] = tweets_labeled["categoria"].values
frequencies = tweets_labeled_categorical['source'].value_counts()

condition = frequencies<100   # you can define it however you want
mask_obs = frequencies[condition].index
mask_dict = dict.fromkeys(mask_obs, 'others')

tweets_labeled_categorical['source'] = tweets_labeled_categorical['source'].replace(mask_dict)
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(categorical_features, ax.flatten()):
    sns.countplot(data = tweets_labeled_categorical, x = variable, hue = 'categoria', ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)