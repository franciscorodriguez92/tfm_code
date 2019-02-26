# https://www.nltk.org/api/nltk.tokenize.html
from nltk.tokenize import TweetTokenizer
import pandas as pd
import os

#tknzr = TweetTokenizer()
#s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
#tknzr.tokenize(s0)

#%% Tokenizador
path = os.getcwd()
df = pd.read_csv(path+'\data\MW_corpus_machismo_frodriguez_2018-12-08.csv')
df_sample = df.sample(100)

#%% Tokenizador
df_sample['tokens'] = df_sample['text'].apply(
        lambda row: TweetTokenizer().tokenize(row.replace('\n',' ') ))
df_sample['text2'] = df_sample['text'].apply(
        lambda row: row.replace('\n',' ') )
df_sample_tokens = df_sample[['status_id','text2','tokens']]
df_sample_tokens.to_csv('tokens.csv')
