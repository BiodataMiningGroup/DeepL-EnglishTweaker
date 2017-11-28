import nltk
import requests
import numpy as np
import tempfile
from subprocess import call
import sys


BASE_URL = 'https://www.deepl.com/jsonrpc'
JSONRPC_VERSION = '2.0'
DEEPL_METHOD = 'LMT_handle_jobs'
DIFFTOOL = 'opendiff'
LANGUAGES = {
    'auto': 'Auto',
    'DE': 'German',
    'EN': 'English',
    'FR': 'French',
    'ES': 'Spanish',
    'IT': 'Italian',
    'NL': 'Dutch',
    'PL': 'Polish'
}


class TranslationError(Exception):
    def __init__(self, message):
        super(TranslationError, self).__init__(message)


def translate(text, to_lang, from_lang='auto', json=False):
    if text is None:
        raise TranslationError('Text can\'t be None.')
    if to_lang not in LANGUAGES.keys():
        raise TranslationError('Language {} not available.'.format(to_lang))
    if from_lang is not None and from_lang not in LANGUAGES.keys():
        raise TranslationError('Language {} not available.'.format(from_lang))

    jobs = []
    if isinstance(text, (list, tuple, np.ndarray)):
        for i in text:
            jobs.append({
                'kind': 'default',
                'raw_en_sentence': i
            })
    else:
        jobs.append({
                    'kind': 'default',
                    'raw_en_sentence': text
                    })

    parameters = {
        'jsonrpc': JSONRPC_VERSION,
        'method': DEEPL_METHOD,
        'params': {
            'jobs': jobs,
            'lang': {
                'user_preferred_langs': [
                    from_lang,
                    to_lang
                ],
                'source_lang_user_selected': from_lang,
                'target_lang': to_lang
            },
        },
    }

    response = requests.post(BASE_URL, json=parameters).json()

    if 'result' not in response:
        raise TranslationError('DeepL call resulted in a unknown result.')

    translations = response['result']['translations']

    for t in translations:
        if len(translations) == 0 \
                or t['beams'] is None \
                or t['beams'][0]['postprocessed_sentence'] is None:
            raise TranslationError('No translations found.')

    if json:
        return response
    return [t['beams'][0]['postprocessed_sentence'] for t in translations]


def splitIntoBlocks(txt, maxlen=5000):
    tokens = nltk.sent_tokenize(txt)
    ret = []
    tmp = ""
    for i in tokens:
        if (len(i) + len(tmp)) >= maxlen:
            ret.append(tmp)
            tmp = i
        else:
            tmp += i
    ret.append(tmp)
    return ret


f = open(sys.argv[1], 'r')
txt = f.read()
f.close()
blocks = splitIntoBlocks(txt, 400)
ret = translate(translate(blocks, from_lang='EN', to_lang='DE'), from_lang="DE", to_lang="EN")
translation = " ".join(ret)
toktxt = nltk.sent_tokenize(txt)
toktranslation = nltk.sent_tokenize(translation)


a = tempfile.NamedTemporaryFile()
for i in toktxt:
    a.write(i + "\n")
b = tempfile.NamedTemporaryFile()
for i in toktranslation:
    b.write(i + "\n")
a.flush()
b.flush()
call([DIFFTOOL, a.name, b.name])
a.close()
b.close()
