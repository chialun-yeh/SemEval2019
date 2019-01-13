import re
import pandas as pd
from nltk import word_tokenize
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def read_glove(dim):
    df = pd.read_csv('../data/glove.6B.' + str(dim) + 'd.txt', sep=" ", quoting=3, header=None, index_col=0)
    glove = {key: val.values for key, val in df.T.items()}
    return glove
    
def cleanText(text):
    # clean quotations
    text = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", text)
    text = re.sub(r'[„“]|(\'\')|(,,)', '"', text)
    # remove URLs
    text = re.sub(r'(www\S+)|(http\S+)|(href)', '', text)
    # remove anything within {} or [] or ().
    text = re.sub(r'\{[^}]*\}|\[[^]]*\]|\([^)]*\)', '', text)
    # remove irrelevant news usage
    text = re.sub(r'Getty [Ii]mages?|Getty|[Ff]ollow us on [Tt]witter|MORE:', '', text)
    # remove @ or # tags or amp or weird ......
    text = re.sub(r'@\S+|#\S+|\&amp|\.{2,}', '', text)
    # remove multiple white spaces
    text = re.sub(r' {2,}', ' ', text)
    # remove newline in the beginning of the file
    text = text.lstrip().replace('\n','')
    return text

def customTokenize(text):
    '''
    lower, strip numbers and punctuation, remove stop words
    '''
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    words = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words