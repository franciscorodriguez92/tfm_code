# -*- coding: utf-8 -*-
#%% imports
import pandas as pd
import os
import src.utils as utils
import src.classifier as clf
import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, cross_val_predict
from src.preprocess import TextCleaner
from src.preprocess import ColumnSelector
from src.preprocess import TypeSelector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

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

#tweets_labeled = tweets_labeled.loc[80:100,:]

#%% 
categorical_features = ['source', 'respuesta', 'respuesta_screen_name',
          'hastag_presence', 'url_presence',
          'media_type', 'mentions_presence', 'verified']
for f in categorical_features:
    tweets_labeled[f] = tweets_labeled[f].astype("category")

#%% 
x_cols = ['source', 'display_text_width', 'respuesta', 'respuesta_screen_name',
          'favorite_count', 'retweet_count', 'hastag_presence',
          'url_presence', 'media_type', 'mentions_presence',
          'followers_count', 'friends_count', 'listed_count', 'statuses_count',
          'favourites_count', 'verified']
classifier = 'logistic_regression'
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
                                          norm=None, min_df=0.01, ngram_range=(1,1)))
])  
    
classifier_pipeline = Pipeline([('feature-union', FeatureUnion([('text-features', text_pipeline), 
                               ('other-features', preprocess_pipeline)
                              ])),
                          ('clf', clf.get_classifier(classifier))
                          ])

#classifier_pipeline.fit(tweets_labeled[x_cols2], tweets_labeled['categoria'])

#%% Pipeline para baseline    
baseline_pipeline = Pipeline([('text_pipeline', text_pipeline),
                          ('clf', clf.get_classifier(classifier))
                          ])
    
#%% Cross validation baseline
# =============================================================================
# print(cross_val_score(baseline_pipeline, tweets_labeled[x_cols2], tweets_labeled['categoria'], cv = 10, n_jobs = 1))
# 
# =============================================================================
#%% Cross validation
# =============================================================================
# print(cross_val_score(classifier_pipeline, tweets_labeled[x_cols2], tweets_labeled['categoria'], cv = 10, n_jobs = 1))
# 
# =============================================================================


#%% métricas de calidad: accuracy, precision, recall, f1
# =============================================================================
# scoring = {'acc': 'accuracy',
#            'precision': 'precision_macro',
#            'recall': 'recall_macro',
#            'f1': 'f1_macro'
#            }
# print(cross_validate(classifier_pipeline, tweets_labeled[x_cols2], tweets_labeled['categoria'], cv = 10, n_jobs = -1, scoring=scoring))
# =============================================================================

#%% Matriz de confusión
#predicted = classifier_pipeline.predict(texto_prueba.drop('categoria', axis=1))
#print np.mean(predicted == texto_prueba['categoria']) 

# =============================================================================
# y_pred = cross_val_predict(classifier_pipeline, tweets_labeled[x_cols2], tweets_labeled['categoria'], cv=10)
# unique_label = np.unique(tweets_labeled['categoria'])
# print(pd.DataFrame(confusion_matrix(tweets_labeled['categoria'], y_pred, labels=unique_label), 
#                    index=['true:{:}'.format(x) for x in unique_label], 
#                    columns=['pred:{:}'.format(x) for x in unique_label]))
# =============================================================================

#%% GridSearchCV

# para chequear los parámetros:: classifier_pipeline.get_params().keys()

# =============================================================================
# import time
# start = time.time()
# parameters = utils.get_grid_parameters(classifier)
# 
# model = GridSearchCV(classifier_pipeline, param_grid=parameters, cv=5,
#                          scoring='accuracy', verbose=1, n_jobs = -1)
# 
# model.fit(tweets_labeled[x_cols2], tweets_labeled['categoria'])
# print("Best score: %0.3f" % model.best_score_)
# print("Best parameters set:")
# best_parameters = model.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
# 
# end = time.time()
# print(end - start)
# 
# =============================================================================
#%% Método de evaluación: 
#1. Hacer 10 repartos diferentes y aleatorios de training y test (70/30) -> ShuffleSplit
#2. Para cada reparto, 
#	2.1 hacer GridSearchCV con el  70%, coger parámetros óptimos
#	2.2 hacer un cross_validate (k=10) con el 30% restante y los parámetros óptimos
#
#Problema: Los parámetros pueden cambiar en cada caso!?
import time
start = time.time()
scoring = {'acc': 'accuracy',
           'precision': 'precision_macro',
           'recall': 'recall_macro',
           'f1': 'f1_macro'
           }

parameters = utils.get_grid_parameters(classifier)
model = GridSearchCV(classifier_pipeline, param_grid=parameters, cv=5,
                         scoring='accuracy', verbose=1, n_jobs = -1)
from sklearn.model_selection import ShuffleSplit

scores_fold = []
train_test_split = ShuffleSplit(n_splits=10, test_size=.30, random_state=0)
for train, test in train_test_split.split(tweets_labeled):
    train = tweets_labeled.iloc[train]
    test = tweets_labeled.iloc[test]
    model.fit(train[x_cols2], train['categoria'])
    test_score = dict(cross_validate(model.best_estimator_, test[x_cols2], test['categoria'], cv = 10, n_jobs = -1, scoring=scoring))
    scores_fold.append(test_score)
    print(test_score)

keys = set([key for i in scores_fold for key,value in i.iteritems()])
scores = {}
for j in keys:
    means = [np.mean(i[j]) for i in scores_fold]
    scores[j] = np.mean(means)

print(scores)    

end = time.time()
print(end - start)