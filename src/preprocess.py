# -*- coding: utf-8 -*-
import re
import string
import emoji
import unidecode


class TextCleaner(object):

    def __init__(self, filter_users=False, filter_hashtags=False,
                 filter_urls=False, convert_hastags=False,
                 lowercase=False, replace_exclamation=False,
                 replace_interrogation=False,
                 remove_accents=False,
                 remove_punctuation=False,
                 replace_emojis=False):  
        self.filter_users = filter_users
        self.filter_hashtags = filter_hashtags
        self.filter_urls = filter_urls
        self.convert_hastags = convert_hastags
        self.lowercase = lowercase
        self.replace_exclamation = replace_exclamation
        self.replace_interrogation = replace_interrogation
        self.remove_accents = remove_accents
        self.remove_punctuation = remove_punctuation
        self.replace_emojis = replace_emojis

    def __call__(self, text):
        #text = text.decode('utf-8')
        if self.replace_emojis:
            text = self.replace_emoji(text)  
        if self.filter_urls:
            text = self.replace_url(text)  
        if self.remove_accents:
            text = self.strip_accents(text)        
        if self.filter_users:
            text = self.replace_user(text)
        if self.convert_hastags:
            text = self.convert_hastags_upper(text)             
        if self.filter_hashtags:
            text = self.replace_hastags(text)          
        if self.replace_exclamation:
            text = self.replace_exclamations(text) 
        if self.replace_interrogation:
            text = self.replace_interrogations(text)
        if self.remove_punctuation:
            text = self.filter_punctuation(text)  
        if self.lowercase:
            text = self.to_lower_endline(text)    
        return text
    

    def to_lower_endline(self, text):
        return text.lower().replace('\n',' ')
    
    def filter_punctuation(self, text):
        #return text.translate(None, string.punctuation)
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def strip_accents(self, text):
        return unidecode.unidecode(text)

    def replace_user(self, text):
        return re.sub( r"(?<=^|(?<=[^a-zA-Z0-9-\\.]))@([A-Za-z]+[A-Za-z0-9]+)", r"twuser", text)
    
    def replace_hastags(self, text):
        return re.sub( r"(?<=^|(?<=[^a-zA-Z0-9-\\.]))#([A-Za-z]+[A-Za-z0-9]+)", r"twhastag", text)
    
    def convert_hastags_upper(self, text):
        return re.sub( r"([A-Z])", r" \1", text)
    
    def replace_exclamations(self, text):
        return re.sub( r"(!+|¡+)", r" twexclamation", text)
    
    def replace_url(self, text):
        return re.sub( r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", r"twurl", text)
    
    def replace_interrogations(self, text):
        text = text.replace(u'¿', 'twinterrogation ')
        return re.sub( r"(\?+)", r" twinterrogation", text)
    
    def replace_emoji(self, text):
        return emoji.demojize(unicode(text)).replace('::',' ')