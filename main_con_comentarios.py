# https://www.nltk.org/api/nltk.tokenize.html
from nltk.tokenize import TweetTokenizer
import pandas as pd
import os

#tknzr = TweetTokenizer()
#s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
#tknzr.tokenize(s0)
#%% imports
from nltk.tokenize import TweetTokenizer
import pandas as pd
import os
import src.preprocess_utils as utils

#%% Read files
path = os.getcwd()
labels = pd.read_table(path + '/resources/data/corpus_machismo_etiquetas.csv', sep=";")
labels = labels[["status_id","categoria"]]

tweets_fields = pd.read_csv(path + '/resources/data/corpus_machismo_frodriguez_atributos_extra.csv', 
                            dtype={'status_id': 'str'})
#status = labels.status_id
#status_ = [x for x in status.astype(str).tolist() if x not in tweets_labeled.status_id.astype(str).tolist()]
#%% Cruce de los ficheros

tweets_fields = utils.change_dtypes(tweets_fields, {'status_id': str})
labels = utils.change_dtypes(labels, {'status_id': str})
tweets_labeled = tweets_fields.merge(labels, on = 'status_id', how = 'inner')

texto_prueba = tweets_labeled.loc[1240:1250, 'text']




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


#%% 

import src.classifier as classifier
import src.textmodel as textmodel
import src.preprocess_utils 
a = classifier.prueba()
textmodel.prueba()
src.preprocess_utils.get_tweet()

#https://github.com/WillKoehrsen/feature-selector
#TODO:
#https://regex101.com/#python
#-Convertir a minúsculas
#-Reemplazar exclamaciones concatenadas por una palabra clave ((!+) (¡+) exclamaciones)
#-Reemplazar interrogaciones concatenadas por una palabra clave(\?+) (\¿+)
#-Reemplazar interrogaciones y exclamaciones concatenadas por una palabra clave -> OK con lo anterior
#-Atributo que detecta presencia de URLs -> OK
#-Reemplazar menciones por una palabra clave (USER) ((?<=^|(?<=[^a-zA-Z0-9-\\.]))@([A-Za-z]+[A-Za-z0-9]+)) 
#-Reemplazar hastags por palabra clave ( (?<=^|(?<=[^a-zA-Z0-9-\\.]))#([A-Za-z]+[A-Za-z0-9]+) )
#-Stemming
#-Ver los hastags con mayúsculas #NosVamosDeFiesta -> Nos vamos de fiesta (re.sub( r"([A-Z])", r" \1", s))
#https://stackoverflow.com/questions/2277352/split-a-string-at-uppercase-letters
#-Lexicón de abreviaturas, por ejemplo TQM por te quiero mucho (http://www.rcs.cic.ipn.mx/2016_115/Compilacion%20de%20un%20lexicon%20de%20redes%20sociales%20para%20la%20identificacion%20de%20perfiles%20de%20autor.pdf)
#-Sustituir emoticonos por palabra clave (crear léxico)
#emoji.demojize(unicode(texto_prueba.iloc[6].decode('utf-8')))

#-Sustituir hastags sexistas por palabra clave (léxico sexista de italia) (Hurtlex)

#.
#├── resources                    # Léxicos, ficheros de entrada, etc.
#│   ├── benchmarks          
#│   ├── json         
#│   └── csv   
#├── src                    # Funciones
#│   ├── __init.py__          
#│   ├── preprocess.py  
#│   └── classifier.py  
#├── test                    # Test files (alternatively `spec` or `tests`)
#└── main.py


#sys.path.append('../lambda')
#import cluster_config
#
#if __name__ == "__main__":
#  parser = argparse.ArgumentParser(description='Myconfig')
#  parser.add_argument('-s', type=str, choices=["dev","int"],help="Stage",required=True)
#  parser.add_argument('--cores', type=int, help="Number of core instances", default=1)
#  parser.add_argument('--spot', action='store_true', help="Core instances will be spot instances")
#  parser.add_argument('--master', nargs='+', default=["c5.xlarge"], help="Master instance type")
#  parser.add_argument('--core', nargs='+', default=["c5.xlarge"], help="Core instances type")
#  parser.add_argument('--key', default=argparse.SUPPRESS, help="EC2 key name for cluster instances, default to networkanalytics-<stage>")
#  parser.add_argument('--profile', default=argparse.SUPPRESS, help="aws cli profile to use")
#  parser.add_argument('--spot-price', default=0.15,help="Price for spot instances")
#  args = parser.parse_args()



#Train_Textlist, Test_Textlist = train_test_split(df_sample_text,
#test_size= 0.3, random_state=13)






#%% Tokenizador

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
#%% 

#https://github.com/WillKoehrsen/feature-selector
#TODO:
#https://regex101.com/#python
#-Convertir a minúsculas
#-Reemplazar exclamaciones concatenadas por una palabra clave ((!+) (¡+) exclamaciones)
#-Reemplazar interrogaciones concatenadas por una palabra clave(\?+) (\¿+)
#-Reemplazar interrogaciones y exclamaciones concatenadas por una palabra clave -> OK con lo anterior
#-Atributo que detecta presencia de URLs -> OK
#-Reemplazar menciones por una palabra clave (USER) ((?<=^|(?<=[^a-zA-Z0-9-\\.]))@([A-Za-z]+[A-Za-z0-9]+)) 
#-Reemplazar hastags por palabra clave ( (?<=^|(?<=[^a-zA-Z0-9-\\.]))#([A-Za-z]+[A-Za-z0-9]+) )
#-Stemming
#-Ver los hastags con mayúsculas #NosVamosDeFiesta -> Nos vamos de fiesta (re.sub( r"([A-Z])", r" \1", s))
#https://stackoverflow.com/questions/2277352/split-a-string-at-uppercase-letters
#-Sustituir emoticonos por palabra clave (crear léxico)
#emoji.demojize(unicode(texto_prueba.iloc[6].decode('utf-8')))

#https://github.com/pan-webis-de/daneshvar18/blob/master/pan18ap/process_data_files.py en load_flame_dictionary carga el fichero en un dict
# replace: https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
#-Lexicón de abreviaturas, por ejemplo TQM por te quiero mucho (http://www.rcs.cic.ipn.mx/2016_115/Compilacion%20de%20un%20lexicon%20de%20redes%20sociales%20para%20la%20identificacion%20de%20perfiles%20de%20autor.pdf)
#-Sustituir hastags sexistas por palabra clave (léxico Hurtlex) (http://ceur-ws.org/Vol-2253/paper49.pdf) (http://hatespeech.di.unito.it/resources.html)


#-Revisar OOV -> De momento no
#-Detección de influencia (se insulta a personas más seguidas) *** -> Ya esta con campos de rtweet

#-Atributos tf-idf


#.
#├── resources                    # Léxicos, ficheros de entrada, etc.
#│   ├── benchmarks          
#│   ├── json         
#│   └── csv   
#├── src                    # Funciones
#│   ├── __init.py__          
#│   ├── preprocess.py  
#│   └── classifier.py  
#├── test                    # Test files (alternatively `spec` or `tests`)
#└── main.py


#sys.path.append('../lambda')
#import cluster_config
#
#if __name__ == "__main__":
#  parser = argparse.ArgumentParser(description='Myconfig')
#  parser.add_argument('-s', type=str, choices=["dev","int"],help="Stage",required=True)
#  parser.add_argument('--cores', type=int, help="Number of core instances", default=1)
#  parser.add_argument('--spot', action='store_true', help="Core instances will be spot instances")
#  parser.add_argument('--master', nargs='+', default=["c5.xlarge"], help="Master instance type")
#  parser.add_argument('--core', nargs='+', default=["c5.xlarge"], help="Core instances type")
#  parser.add_argument('--key', default=argparse.SUPPRESS, help="EC2 key name for cluster instances, default to networkanalytics-<stage>")
#  parser.add_argument('--profile', default=argparse.SUPPRESS, help="aws cli profile to use")
#  parser.add_argument('--spot-price', default=0.15,help="Price for spot instances")
#  args = parser.parse_args()

