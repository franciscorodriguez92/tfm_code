import hunspell


class HunspellWrapper(object):

    def __init__(self, lang='en'):
        self.hobj = hunspell.HunSpell(
                '/root/hunspell/es_ANY.dic', '/root/hunspell/es_ANY.aff')

    def is_correct(self, text):
        return self.hobj.spell(text)

    def get_suggestion(self, text):
        return self.hobj.suggest(text)[0]