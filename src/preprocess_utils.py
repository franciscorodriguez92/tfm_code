# -*- coding: utf-8 -*-

import string
import pandas as pd
from nltk.corpus import stopwords
from nltk import PorterStemmer
import re
import emoji
import os
from nltk.tokenize import TweetTokenizer
import unidecode


def to_lower_endline(text):
    return text.lower().replace('\n',' ')

def change_dtypes(df, convert_dict):
    return df.astype(convert_dict)

def remove_punctuation(text):
    return text.translate(None, string.punctuation)

def remove_accents(text):
    return unidecode.unidecode(text)

def replace_user(text):
    return re.sub( r"(?<=^|(?<=[^a-zA-Z0-9-\\.]))@([A-Za-z]+[A-Za-z0-9]+)", r"twuser", text)

def replace_hastags(text):
    return re.sub( r"(?<=^|(?<=[^a-zA-Z0-9-\\.]))#([A-Za-z]+[A-Za-z0-9]+)", r"twhastag", text)

def convert_hastags(text):
    return re.sub( r"([A-Z])", r" \1", text)

def replace_exclamation(text):
    return re.sub( r"(!+|¡+)", r" twexclamation", text)

def replace_url(text):
    return re.sub( r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", r"twurl", text)

def replace_interrogation(text):
    text = text.replace(u'¿', 'twinterrogation ')
    return re.sub( r"(\?+)", r" twinterrogation", text)

def replace_emoji(text):
    return emoji.demojize(unicode(text))

#Requieren tokenización:
def filter_stopwords(word_list):
    filtered_words = [word for word in word_list if word not in stopwords.words('spanish')]
    return filtered_words

def stemming(word_list):
    filtered_words_stem=[PorterStemmer().stem(word) for word in word_list]
    return filtered_words_stem

def replace_abb(tokens):
    path = os.getcwd()
    columns = ["word","label"]
    slang = pd.read_table(path + '/resources/lexicon/SP/SPslang.txt', names=columns, header=None, index_col=False)
    slang['word'] = slang['word'].str.decode("utf-8")
    slang['label'] = slang['label'].str.decode("utf-8")
    slang_dict = slang.set_index('word')['label'].to_dict()
    rep = dict((re.escape(k), v) for k, v in slang_dict.items())
    return replace(tokens, rep)

def replace(list, dictionary):
    new_list = []

    for i in list:
        if i in dictionary:
            new_list.append(dictionary[i])
        else:
          new_list.append(i)

    return new_list
          

def tokenizer_(text):
    tokens = TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(
            unidecode.unidecode(unidecode.unidecode(text)))
    tokens = stemming(replace_abb(filter_stopwords(tokens)))
    return tokens


#def replace_hurtlex(tokens):
#    path = os.getcwd()
#    columns = ["category","stereotype", "word"]
#    hurtlex_conservative = pd.read_table(path + '/resources/hurtlex/hurtlex/hurtlex_ES_conservative.tsv', names=columns, header=None, index_col=False)
#    hurtlex_conservative = hurtlex_conservative.loc[(hurtlex_conservative['category'] == 'asf') | (hurtlex_conservative['category'] <= 'pr')]
#    hurtlex_conservative = hurtlex_conservative['word']
#    hurtlex_inclusive = pd.read_table(path + '/resources/hurtlex/hurtlex/hurtlex_ES_inclusive.tsv', names=columns, header=None, index_col=False)
#    hurtlex_inclusive = hurtlex_inclusive.loc[(hurtlex_inclusive['category'] == 'asf') | (hurtlex_inclusive['category'] <= 'pr')]
#    hurtlex_inclusive = hurtlex_inclusive['word']
#    hurtlex = set(hurtlex_conservative + hurtlex_inclusive)
#    return(list(set(tokens) & set(hurtlex)))

