import re
import pandas as pd
import numpy as np
import string
from lxml import etree
import nltk
from nltk.corpus import stopwords
import html
from sklearn.model_selection import StratifiedShuffleSplit

'''Contains some text cleaning functions'''

def cleanQuotations(text):
    # clean quotations
    text = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", text)
    text = re.sub(r'[„“”]|(\'\')|(,,)', '"', text)
    return text

def cleanText(text):  
    # remove URLs
    text = re.sub(r'(www\S+)|(https?\S+)|(href)', ' ', text)
    # remove anything within {} or [] or ().
    text = re.sub(r'\{[^}]*\}|\[[^]]*\]|\([^)]*\)', ' ', text)
    # remove irrelevant news usage
    text = re.sub(r'Getty [Ii]mages?|Getty|[Ff]ollow us on [Tt]witter|MORE:|ADVERTISEMENT|VIDEO', ' ', text)
    # remove @ or # tags or weird ......
    text = re.sub(r'@\S+|#\S+|\.{2,}', ' ', text)
    # remove newline in the beginning of the file
    text = text.lstrip().replace('\n','')
    # remove multiple white spaces
    re1 = re.compile(r'  +')
    text = re1.sub(' ', text)
    return text

def fixup(text):
    '''
    fix some HTML codes and white spaces (from Jeremy Howard)
    '''
    text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ') \
    .replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"')\
    .replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').replace('\\', ' \\ ')
    return html.unescape(text)

def read_glove(path, dim):
    '''
    read the glove vectors
    dim: 100, 200, 300
    '''
    df = pd.read_csv(path + 'glove.6B.' + str(dim) + 'd.txt', sep=" ", quoting=3, header=None, index_col=0)
    glove = {key: val.values for key, val in df.T.items()}
    return glove

def customTokenize(text, rm_stopwords=False):
    '''
    lower, strip numbers and punctuation, remove stop words
    '''
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    words = [word for word in tokens if word.isalpha()]
    if rm_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
    return words

def fixedTestSplit(labels):
    '''
    split into training and held-out test set with balanced class
    '''
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state = 1)
    split_idx = list(sss.split(np.zeros(len(labels)), labels))[0]
    return split_idx[0], split_idx[1]
