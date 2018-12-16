import os
import numpy as np
from pathlib import Path
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import SaveLoad, simple_preprocess
from gensim.matutils import corpus2csc
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, strip_non_alphanum, strip_multiple_whitespaces, strip_numeric, strip_short
from six import iteritems


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
        # no need to filter terms here because only terms that exist in the dictionary count
        filters = [strip_non_alphanum, strip_multiple_whitespaces]
        for f in self.file:
            if self.stem:
                p = PorterStemmer()
                for line in open(f, 'r', encoding='utf-8'):
                    yield self.dictionary.doc2bow(preprocess_string(p.stem_sentence(line.split(',')[1]), filters).split())
            else:
                for line in open(f, 'r', encoding='utf-8'):
                    yield self.dictionary.doc2bow(preprocess_string(line.split(',')[1], filter).split())


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
            for line in open(f, 'r', encoding='utf-8'):
                docId = line.split(',')[0]
                text = line.split(',')[1]
                yield TaggedDocument(simple_preprocess(text), [docId])

def computeTFIDF(inputFileList, dictionary, file_path, stem):
    if os.path.exists(file_path + 'tfidf_model'):
        return
    else:
        print('computing TFIDF...')
        bow = Bow(inputFileList, dictionary, stem)
        model = models.TfidfModel(bow)
        print('saving tfidf model...')
        model.save(file_path + 'tfidf_model')

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

def buildRep(inputFileFolder, rep='bow', dim=50000, stem=False):
    '''
    Build document representation model

    InputFileFolder: path of the input file folder
    rep: representation (bow, tfidf or doc2vec)
    dim: dimension of the representation
    '''
    file_path = 'tmp/'
    fileList = [inputFileFolder + f for f in os.listdir(inputFileFolder) if f.endswith(".txt")]
    dct_name = 'no_stem'

    if rep == 'bow' or rep =='tfidf':
        if os.path.exists( file_path + dct_name + '.dict'):
            print('loading bow dictionary...')
            dct = corpora.Dictionary.load(file_path + dct_name + '.dict')
        else:
            print('building bow dictionary...')
            filters = [lambda x: x.lower(), remove_stopwords, strip_non_alphanum, strip_multiple_whitespaces, strip_numeric, strip_short]
            if stem:
                p = PorterStemmer()
                dct = corpora.Dictionary(preprocess_string(p.stem_sentence(line.split(',')[1]), filters) for line in open(fileList[0], 'r', encoding='utf8')) 
                for f in fileList[1:]:
                    dct.add_documents(preprocess_string(p.stem_sentence(line.split(',')[1]), filters) for line in open(f, 'r', encoding='utf8')) 
            else:
                dct = corpora.Dictionary(preprocess_string(line.split(',')[1], filters) for line in open(fileList[0], 'r', encoding='utf8')) 
                for f in fileList[1:]:
                    dct.add_documents(preprocess_string(line.split(',')[1], filters) for line in open(f, 'r', encoding='utf8')) 
            # filter terms that occur only once
            once_ids = [tokenid for tokenid, docfreq in iteritems(dct.dfs) if docfreq == 1]
            dct.filter_tokens(once_ids)
            dct.filter_extremes(no_below=10, no_above=0.1, keep_n=dim)
            dct.compactify()
            print('saving bow dictionary....')
            dct.save(file_path + dct_name + '.dict')

        computeTFIDF(fileList, dct, file_path, stem)

    else:
        computeDoc2Vec(fileList, file_path, dim)


def extract_doc_rep(textFile, rep='bow', dim=50000, stem=False):
    '''
    Extract the representation for file docFile
    '''
    file_path = 'tmp/'
    dct_name = 'no_stem'
    file_name = Path(textFile).name.strip('txt')
    if rep =='bow' or rep =='tfidf':
        if os.path.exists(file_path + file_name + '.mm'):
            bow = corpora.MmCorpus(file_path + file_name + '.mm')
        else:
            dct = corpora.Dictionary.load(file_path + dct_name + '.dict')   
            bow = Bow(textFile, dct, stem)
            corpora.MmCorpus.serialize(file_path + file_name + '.mm', bow)  
        if rep == 'bow':
            return corpus2csc(bow, num_terms=dim)
        else:
            print('loading tfidf model...')
            tfidfmodel = SaveLoad.load(os.path.join(file_path, 'tfidf_model'))
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
            tag = line.split(',')[0]      
            #text = line.split(',')[1]
            vectors[:,cnt] = model[tag]
            #vectors[:,cnt] = model.infer_vector(simple_preprocess(text))
            cnt = cnt+1
        return vectors
        






