import os
import numpy as np
from pathlib import Path
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import SaveLoad, simple_preprocess
from gensim.matutils import corpus2csc
from gensim.parsing.preprocessing import stem_text, remove_stopwords, strip_multiple_whitespaces
from gensim.parsing.preprocessing import preprocess_string, strip_non_alphanum, strip_numeric, strip_short

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer 
import string
import nltk

class Bow(object):
    '''
    compute Bag-of-Word vectors using iterator
    '''
    def __init__(self, fileList, dct, stem):
        self.file = fileList
        self.dictionary = dct
        self.stem = stem
        
    def __iter__(self):
        if isinstance(self.file, str):
            self.file = [self.file]
        for f in self.file:
            for line in open(f, encoding='utf-8'):
                text = line.split('::')[1] + ' ' + line.split('::')[2]
                yield self.dictionary.doc2bow( preprocess(text, self.stem) )

class TaggedDoc(object):
    '''
    prepare tagged documents for training the doc2vec model
    '''
    def __init__(self, fileList):
        self.file = fileList

    def __iter__(self):
        if isinstance(self.file, str):
            self.file = [self.file]
        for f in self.file:
            for line in open(f, encoding='utf-8'):
                docId = line.split('::')[0]
                text = line.split('::')[1] + ' ' + line.split('::')[2]
                yield TaggedDocument(simple_preprocess(text), [docId])

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatizeText(text):
    # Init Lemmatizer
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text) if w not in string.punctuation])

def buildName(stem, model_name):
    if stem == 1:
        return model_name + '_stem'
    elif stem == 2:
        return model_name + '_lemmatize'
    else:
        return model_name

def computeTFIDF(inputFileList, dictionary, file_path, stem):
    '''Build tfidf model'''
    if os.path.exists(file_path + buildName(stem, 'tfidf_model') ):
        return
    else:
        print('computing TFIDF...')
        bow = Bow(inputFileList, dictionary, stem)
        model = models.TfidfModel(bow, dictionary)
        print('saving tfidf model...')
        model.save(file_path + buildName(stem, 'tfidf_model'))

def computeDoc2Vec(inputFileList, file_path, dim):
    '''Build doc2vec model'''
    if os.path.exists(file_path + 'doc2vec'):
        return
    else:
        print('computing doc2vec...')
        train_corpus = TaggedDoc(inputFileList)
        # training
        model = Doc2Vec(train_corpus, vector_size=dim, min_count=1, epochs=20)
        print('saving doc2vec model...')
        model.save(file_path + 'doc2vec')

def preprocess(text, stem):
    '''
    return a list of tokenized words filtered by filters
    '''
    #filters = [lambda x: x.lower(), remove_stopwords, strip_non_alphanum, strip_multiple_whitespaces, strip_numeric, strip_short]
    # return simple_preprocess(lemmatizeText(text), True) if stem else simple_preprocess(text, True)
    if stem == 1:
        return simple_preprocess(stem_text(text), True)
    elif stem == 2:
        return simple_preprocess(lemmatizeText(text), True)
    else:
        return simple_preprocess(text, True)
    

def buildRep(inputFileFolder, rep='bow', dim=50000, stem=False):
    '''
    Build document representation model

    InputFileFolder: path of the input file folder
    rep: representation (bow, tfidf or doc2vec)
    dim: dimension of the representation
    '''
    file_path = 'tmp/'
    fileList = [inputFileFolder + f for f in os.listdir(inputFileFolder) if f.startswith('articles') and f.endswith("bypublisher.txt")]

    if rep == 'bow' or rep =='tfidf':
        if os.path.exists( file_path + buildName(stem, 'dictionary') + '.dict'):
            print('loading bow dictionary...')
            dct = corpora.Dictionary.load(file_path + buildName(stem, 'dictionary') + '.dict')
        else:
            print('building bow dictionary...')
            dct = corpora.Dictionary( preprocess(line.split('::')[1] + ' ' + line.split('::')[2], stem) \
            for line in open(fileList[0], encoding='utf8') )
            for f in fileList[1:]:
                dct.add_documents( preprocess(line.split('::')[1] + ' ' + line.split('::')[2], stem) \
                for line in open (f, encoding='utf8') )                     
            
            # filter terms that occur in less than 10 documents or more than 50% of the documents and keeps only the first dim frequent words
            dct.filter_extremes(no_below=5, no_above=0.5)
            #dct.filter_extremes(no_below=10, no_above=0.5, keep_n=dim)
            dct.compactify()
            
            # save dictionary
            print('saving bow dictionary....')
            dct.save(file_path + buildName(stem, 'dictionary') + '.dict')

        computeTFIDF(fileList, dct, file_path, stem)

    else:
        computeDoc2Vec(fileList, file_path, dim)


def extract_doc_rep(textFile, rep='bow', dim=50000, stem=False):
    '''
    Extract the representation for file docFile
    '''
    file_path = 'tmp/'
    file_name = Path(textFile).name.strip('txt')

    if rep =='bow' or rep =='tfidf':
        if os.path.exists(file_path + buildName(stem, file_name) + '.mm'):
            bow = corpora.MmCorpus(file_path + buildName(stem, file_name) + '.mm')
        else:
            dct = corpora.Dictionary.load(file_path + buildName(stem, 'dictionary') + '.dict')   
            bow = Bow(textFile, dct, stem)
            corpora.MmCorpus.serialize(file_path + buildName(stem, file_name) + '.mm', bow)  
        if rep == 'bow':
            return corpus2csc(bow, num_terms=dim)
        else:
            print('loading tfidf model...')
            tfidfmodel = SaveLoad.load(file_path + buildName(stem, 'tfidf_model'))
            return corpus2csc(tfidfmodel[bow], num_terms=dim)

    elif rep == 'doc2vec':
        numDoc = 0
        for line in open(textFile, encoding='utf8'):
            numDoc = numDoc + 1
        print('loading doc2vec model...')
        model = Doc2Vec.load(file_path + 'doc2vec')
        # infer vectors
        vectors = np.zeros((dim, numDoc))
        cnt = 0
        for line in open(textFile, encoding='utf8'):  
            tag = line.split('::')[0]      
            vectors[:, cnt] = model[tag]
            #vectors[:,cnt] = model.infer_vector(simple_preprocess(text))
            cnt = cnt+1
        return vectors
        


if __name__ == "__main__":
     buildRep('../data/', rep='bow', dim=50000, stem=2)

