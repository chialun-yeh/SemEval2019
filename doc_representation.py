import os
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import SaveLoad
from gensim.matutils import corpus2csc
from six import iteritems
from nltk.corpus import stopwords


class Bow(object):
    '''
    compute Bag-of-Word vectors using iterator
    '''
    def __init__(self, data, dic):
        self.file = data
        self.dictionary = dic
        
    def __iter__(self):
        if isinstance(self.file, str):
            self.file = [self.file]
        for f in self.file:
            for line in open(f):
                yield self.dictionary.doc2bow(line.split(',')[1].split())

class TaggedDoc(object):
    '''
    prepare tagged documents for training the doc2vec model
    '''
    def __init__(self, data):
        self.file = data
    def __iter__(self):
        if isinstance(self.file, str):
            self.file = [self.file]
        for f in self.file:
            for line in open(f):
                yield (TaggedDocument(line.split(',')[1].split(), line.split(',')[0]))

def computeTFIDF(inputFile, dictionary, file_path):
    if os.path.exists( os.path.join(file_path, 'tfidf_model')):
        return
    else:
        print('computing TFIDF...')
        bow = Bow(inputFile, dictionary)
        model = models.TfidfModel(bow)
        print('saving tfidf model...')
        model.save(os.path.join(file_path,'tfidf_model'))

def computeDoc2Vec(inputFile, file_path):
    print('computing doc2vec...')
    documents = TaggedDoc(inputFile)
    model = Doc2Vec(documents, vector_size=300, min_count=2, epochs=15)
    print('saving doc2vec model...')
    model.save(os.path.join(file_path, 'doc2vec'))

def buildRep(inputFile, rep='bow', dim=50000, sample=False):
    '''
    Build document representation model using all documents
    '''
    if sample:
        file_path = 'tmp/sample/'
    else:
        file_path = 'tmp/'

    if rep == 'bow' or rep =='tfidf':
        if os.path.exists( os.path.join(file_path, 'dictionary.dict')):
            print('loading bow dictionary...')
            dictionary = corpora.Dictionary.load(os.path.join(file_path,'dictionary.dict'))   
        else:
            print('building bow dictionary')
            dictionary = corpora.Dictionary(line.split(',')[1].split() for line in open(inputFile[0]))
            if len(inputFile) > 1:
                for f in inputFile[1:]:
                    dictionary.add_documents(line.split(',')[1].split() for line in open(f))          
            stoplist = stopwords.words('english')
            stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
            once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
            dictionary.filter_tokens(stop_ids + once_ids)
            dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=dim)
            dictionary.compactify()
            print('saving bow dictionary....')
            dictionary.save(os.path.join(file_path,'dictionary.dict'))

        computeTFIDF(inputFile, dictionary, file_path)

    else:
        computeDoc2Vec(inputFile, file_path)


def extract_doc_rep(docFile, rep='bow', name='', sample=False):
    '''
    Extract the representation for file docFile
    '''
    if sample:
        file_path = 'tmp/sample/'
    else:
        file_path = 'tmp/'

    if rep =='bow' or rep =='tfidf':
        if os.path.exists(os.path.join(file_path, name + '.mm')):
            bow = corpora.MmCorpus(os.path.join(file_path, name + '.mm'))
        else:
            dictionary = corpora.Dictionary.load(os.path.join(file_path,'dictionary.dict'))   
            bow = Bow(docFile, dictionary)
            corpora.MmCorpus.serialize(os.path.join(file_path, name + '.mm'), bow)  
        if rep == 'bow':
            return corpus2csc(bow, num_terms=50000)
        else:
            print('loading tfidf model...')
            model = SaveLoad.load(os.path.join(file_path, 'tfidf_model'))
            return corpus2csc(model[bow], num_terms=50000)

    elif rep == 'doc2vec':
        print('loading doc2vec model...')
        model = Doc2Vec.load(os.path.join(file_path,'doc2vec'))
        # infer vectors

    else:
        print('invalid representation')
        






