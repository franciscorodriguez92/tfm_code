# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:35:27 2018
@author: Francisco Miguel Rodríguez Sánchez
"""
#%% Imports
import os 
import glob
import pandas as pd
import collections, nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import PorterStemmer
import re

#%% AUX Read function
def read_files(file_path, encoding = None):
    path = os.getcwd()
    file_list = glob.glob(path + file_path)
    corpus=pd.DataFrame()
    list_ = []
    for file_ in file_list:
        df = pd.read_table(file_, encoding = encoding)
        list_.append(df)
    corpus = pd.concat(list_)
    return corpus

#%% Modelo Unigrama y perplejidad
def unigram(tokens):    
    model = collections.defaultdict(lambda: 0.00000001)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    v = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/v
    return model

def unigram_laplace(tokens, alpha=1):    
    model = collections.defaultdict(lambda: 1/float(len(tokens)+len(set(tokens)))) 
    counts = dict()
    for f in tokens:
        try:
            counts[f] += 1
        except KeyError:
            counts [f] = 1
            continue
    for word in counts:
        model[word] = (counts[word]+alpha)/float(sum(counts.values())+len(set(tokens)))
    return model

def perplexity(testset, model):
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

#%% Función para tokenizar con chunks
def get_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text.decode('utf-8').encode('ascii', 'ignore'))))
    continuous_chunk = []
    current_chunk = []
    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continuous_chunk.append(subtree[0])
    if not continuous_chunk:
        continuous_chunk = ['nan'] 
    return continuous_chunk
#%% Stopwords
def filter_stopwords(word_list):
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    return filtered_words

#%% Stopwords and stemming
def filter_stopwords_stem(word_list):
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    filtered_words_stem=[PorterStemmer().stem(word) for word in filtered_words]
    return filtered_words_stem

#%% Calcular métricas
    
def compute_metrics (test_predicciones, umbrales):
    test_predicciones_sorted = test_predicciones.sort_values('perplexity').reset_index()
    n = umbrales
    step_umbrales = test_predicciones_sorted.shape[0]/n
    num_related = sum(test_predicciones_sorted['filtering'] == 'RELATED')
    recall = []
    precision = []
    accuracy = []
    perplexity = []
    for i in range(step_umbrales,
                   test_predicciones_sorted.shape[0] + step_umbrales, 
                   step_umbrales):
        test_predicciones_sorted['pred'] = np.where(
                test_predicciones_sorted.index<=i-1, 'RELATED', 'UNRELATED')
        true_positive = sum( (test_predicciones_sorted[
                'filtering'] == test_predicciones_sorted[
                        'pred']) & (test_predicciones_sorted[
                        'pred'] == 'RELATED')  )
        correct_classification =sum(test_predicciones_sorted[
                'filtering'] == test_predicciones_sorted[
                        'pred'])
        num_related_pred = sum(test_predicciones_sorted['pred'] == 'RELATED')
        recall.append(float(true_positive)/num_related)
        precision.append(float(true_positive)/(num_related_pred))
        accuracy.append(float(correct_classification)/test_predicciones_sorted.shape[0])
        perplexity.append(test_predicciones_sorted['perplexity'][i-1])
    return recall, precision, accuracy, perplexity


#%% Plot curve precision-recall
    
def plot_curve_precision_recall(precision, recall):
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    plt.plot(precision, recall, '-o')
    plt.axis([0, 1.07, 0, 1.07])
    plt.show()

#%% TRAINING
    
#%% Load data training labeled
training_labeled = read_files(file_path='\\data\\training\\labeled\\*',
                              encoding="cp1252")
training_labeled = training_labeled[['tweet_id','filtering']]

#%% Load data training info
training_info = read_files(file_path='\\data\\training\\tweet_info\\*')
training_info = training_info[['tweet_id','tweet_url','language']]
#solo tweets en inglés:
training_info_english = training_info[training_info.language == 'EN']

#%% Load data training text
training_text = read_files(file_path='\\data\\training\\tweet_text\\*')

#%% JOIN de los 3 ficheros training
training_english = training_info_english.merge(training_text.merge(training_labeled, 
                                                           on = 'tweet_id'), 
    on='tweet_id')

#se añade el nombre y categoría de las entidades
entities_dataset = pd.read_table('.\\data\\replab2013_entities.tsv')
entities_dataset = entities_dataset[['entity_id','entity_name','category']]
training_english = training_english.merge(entities_dataset, on = 'entity_id')
training_english_related = training_english[training_english.filtering == 'RELATED']


#%% POR CATEGORIA

#%% Modelos por categoría
training_english_category = training_english_related.groupby('category')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_category['models'] = training_english_category['text'].apply(
        lambda row: unigram(nltk.word_tokenize(row.decode('utf-8'))))

#%% Modelos por categoría con smoothing laplace
training_english_category_laplace = training_english_related.groupby('category')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_category_laplace['models'] = training_english_category_laplace['text'].apply(
        lambda row: unigram_laplace(nltk.word_tokenize(row.decode('utf-8'))))

#%% Modelos por categoría con chunks
training_english_category_chunks = training_english_related.groupby('category')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_category_chunks['models'] = training_english_category_chunks['text'].apply(
        lambda row: unigram(get_chunks(row)))

#%% Modelos por categoría stopwords y mayúsculas
training_english_category_stopwords = training_english_related.groupby('category')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_category_stopwords['models'] = training_english_category_stopwords['text'].apply(
        lambda row: unigram(filter_stopwords(nltk.word_tokenize(row.decode('utf-8').lower()))))

#%% Modelos por categoría stopwords, mayúsculas, stemming y enlaces
training_english_category_stop_stem = training_english_related.groupby('category')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_category_stop_stem['models'] = training_english_category_stop_stem['text'].apply(
        lambda row: unigram(filter_stopwords_stem(nltk.word_tokenize(re.sub(r"http\S+", "",
                row.decode('utf-8').lower())))))

#%% Modelos por categoría chunks, stopwords, mayúsculas
training_english_category_stopwords_chunks = training_english_related.groupby('category')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_category_stopwords_chunks['models'] = training_english_category_stopwords_chunks['text'].apply(
        lambda row: unigram(filter_stopwords(get_chunks(row.lower()))))

#%% POR ENTIDAD

#%% Modelos por entidad
training_english_entity = training_english_related.groupby('entity_name')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_entity['models'] = training_english_entity['text'].apply(
        lambda row: unigram(nltk.word_tokenize(row.decode('utf-8'))))

#%% Modelos por entidad con chunks
training_english_entity_chunks = training_english_related.groupby('entity_name')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_entity_chunks['models'] = training_english_entity_chunks['text'].apply(
        lambda row: unigram(get_chunks(row)))

#%% #%% Modelos por entidad con stopwords y mayúsculas
training_english_entity_stopwords = training_english_related.groupby('entity_name')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_entity_stopwords['models'] = training_english_entity_stopwords['text'].apply(
        lambda row: unigram(filter_stopwords(nltk.word_tokenize(row.decode('utf-8').lower()))))

#%% GLOBAL

#%% Modelo global 
training_english_global = training_english_related.groupby('language')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_global['models'] = training_english_global['text'].apply(
        lambda row: unigram(nltk.word_tokenize(row.decode('utf-8'))))

#%% Modelo global con chunks
training_english_global_chunks = training_english_related.groupby('language')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_global_chunks['models'] = training_english_global_chunks['text'].apply(
        lambda row: unigram(get_chunks(row)))

#%% Modelo global con stopwords y mayúsculas 
training_english_global_stopwords = training_english_related.groupby('language')['text'].apply(
        lambda x: ' '.join(x.astype(str))).reset_index()
training_english_global_stopwords['models'] = training_english_global_stopwords['text'].apply(
        lambda row: unigram(filter_stopwords(nltk.word_tokenize(row.decode('utf-8').lower()))))

#%% TEST

#%% Load data test labeled
#PROBLEMA ENCONTRADO: FORMATO FICHEROS Y CODIFICACION
test_labeled = read_files(file_path='\\data\\test\\labeled\\*',
                          encoding="cp1252")
test_labeled = test_labeled[['tweet_id','filtering']] 

#%% Load data test info
test_info = read_files(file_path='\\data\\test\\tweet_info\\*')
test_info = test_info[['tweet_id','tweet_url','language']]
#solo tweets en inglés
test_info_english = test_info[test_info.language == 'EN']

#%% Load data test text
test_text = read_files(file_path='\\data\\test\\tweet_text\\*')

#%% JOIN de los 3 ficheros test
test_english = test_info_english.merge(test_text.merge(test_labeled, 
                                                           on = 'tweet_id'), 
    on='tweet_id')
test_english = test_english.merge(entities_dataset, on = 'entity_id')

#%% PREDICCIONES  por CATEGORIA

#%% PREDICCIONES para modelos por categoría
#Iteramos sobre todos los valores de categoría, se filtra el test,
#Se calcula la perplejeidad de cada tweet y se concatena la información
list_ = []
test_predicciones_categoria = pd.DataFrame()
for i in test_english.category.unique():
    test = test_english[test_english.category == i]
    model = training_english_category[training_english_category.category == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            nltk.word_tokenize(row.decode('utf-8')),model))
    list_.append(test)
test_predicciones_categoria = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(test_predicciones_categoria, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES para modelos por categoría con smoothing laplace
list_ = []
test_predicciones_categoria_laplace = pd.DataFrame()
for i in test_english.category.unique():
    test = test_english[test_english.category == i]
    model = training_english_category_laplace[
            training_english_category_laplace.category == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            nltk.word_tokenize(row.decode('utf-8')),model))
    list_.append(test)
test_predicciones_categoria_laplace = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(
        test_predicciones_categoria_laplace, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES para modelos por categoría con chunks
list_ = []
test_predicciones_categoria_chunks = pd.DataFrame()
for i in test_english.category.unique():
    test = test_english[test_english.category == i]
    model = training_english_category_chunks[
            training_english_category_chunks.category == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            get_chunks(row),model))
    list_.append(test)
test_predicciones_categoria_chunks = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(
        test_predicciones_categoria_chunks, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES para modelos por categoría stopwords y mayúsculas
list_ = []
test_predicciones_categoria_stopwords = pd.DataFrame()
for i in test_english.category.unique():
    test = test_english[test_english.category == i]
    model = training_english_category_stopwords[
            training_english_category_stopwords.category == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            filter_stopwords(nltk.word_tokenize(row.decode('utf-8').lower())),model))
    list_.append(test)
test_predicciones_categoria_stopwords = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(test_predicciones_categoria_stopwords, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES para modelos stopwords, mayúsculas, stemming y enlaces
list_ = []
test_predicciones_categoria_stem = pd.DataFrame()
for i in test_english.category.unique():
    test = test_english[test_english.category == i]
    model = training_english_category_stop_stem[
            training_english_category_stop_stem.category == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            filter_stopwords_stem(nltk.word_tokenize(re.sub(r"http\S+", "",
                row.decode('utf-8').lower()))),model))
    list_.append(test)
test_predicciones_categoria_stem = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(test_predicciones_categoria_stem, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES para modelos chunks, stopwords, mayúsculas
list_ = []
test_predicciones_categoria_chunks_stop = pd.DataFrame()
for i in test_english.category.unique():
    test = test_english[test_english.category == i]
    model = training_english_category_stopwords_chunks[
            training_english_category_stopwords_chunks.category == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            filter_stopwords(get_chunks(row.lower())),model))
    list_.append(test)
test_predicciones_categoria_chunks_stop = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(test_predicciones_categoria_chunks_stop, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES  por ENTIDAD

#%% PREDICCIONES para modelos por entidad
list_ = []
test_predicciones_entidad = pd.DataFrame()
for i in test_english.entity_name.unique():
    test = test_english[test_english.entity_name == i]
    model = training_english_entity[training_english_entity.entity_name == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            nltk.word_tokenize(row.decode('utf-8')),model))
    list_.append(test)
test_predicciones_entidad = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(test_predicciones_entidad, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES para modelos por entidad con chunks
list_ = []
test_predicciones_entidad_chunks = pd.DataFrame()
for i in test_english.entity_name.unique():
    test = test_english[test_english.entity_name == i]
    model = training_english_entity_chunks[
            training_english_entity_chunks.entity_name == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            get_chunks(row),model))
    list_.append(test)
test_predicciones_entidad_chunks = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(
        test_predicciones_entidad_chunks, puntos_curva)
plot_curve_precision_recall(precision, recall)
#%% PREDICCIONES para modelos por entidad con chunks, stopwords y mayúsculas
list_ = []
test_predicciones_entidad_stopwords = pd.DataFrame()
for i in test_english.entity_name.unique():
    test = test_english[test_english.entity_name == i]
    model = training_english_entity_stopwords[
            training_english_entity_stopwords.entity_name == i]['models'].iloc[0]
    test['perplexity']=test['text'].astype(str).apply(lambda row: perplexity(
            filter_stopwords(nltk.word_tokenize(row.decode('utf-8').lower())),model))
    list_.append(test)
test_predicciones_entidad_stopwords = pd.concat(list_)

#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(
        test_predicciones_entidad_stopwords, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES GLOBAL

#%% PREDICCIONES para modelo global
test_predicciones_global = test_english
model = training_english_global['models'].iloc[0]
test_predicciones_global['perplexity']=test_english['text'].astype(str).apply(lambda row: perplexity(
            nltk.word_tokenize(row.decode('utf-8')),model))
#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(test_predicciones_global, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES para modelo global con chunks
test_predicciones_global_chunks = test_english
model = training_english_global_chunks['models'].iloc[0]
test_predicciones_global_chunks['perplexity']=test_english['text'].astype(str).apply(lambda row: perplexity(
            get_chunks(row),model))
#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(test_predicciones_global_chunks, puntos_curva)
plot_curve_precision_recall(precision, recall)

#%% PREDICCIONES para modelo global con stopwords y mayúsculas 
test_predicciones_global_stopwords = test_english
model = training_english_global_stopwords['models'].iloc[0]
test_predicciones_global_stopwords['perplexity']=test_english['text'].astype(str).apply(lambda row: perplexity(
            filter_stopwords(nltk.word_tokenize(row.decode('utf-8').lower())),model))
#Curva de precisión/recall (23 umbrales de perplexity)
puntos_curva = 23
recall, precision, accuracy, perplexity = compute_metrics(test_predicciones_global_stopwords, puntos_curva)
plot_curve_precision_recall(precision, recall)
