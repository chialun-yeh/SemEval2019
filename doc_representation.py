import os
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import SaveLoad
from gensim.matutils import corpus2csc
from six import iteritems
from nltk.corpus import stopwords


class Bow(object):
    def __init__(self, data, dic):
        self.file = data
        self.dictionary = dic
        
    def __iter__(self):
        if isinstance(self.file, str):
            self.file = [self.file]
        for f in self.file:
            for line in open(f):
                yield self.dictionary.doc2bow(line.split(',')[1].split())

class prepareTaggedDoc(object):
    def __init__(self, data):
        self.file = data
    def __iter__(self):
        if isinstance(self.file, str):
            self.file = [self.file]
        for f in self.file:
            for line in open(f):
                yield (TaggedDocument(line.split(',')[1].split(), line.split(',')[0]))

def createDict(inputFile, dim=50000):
    if os.path.exists('tmp/dictionary.dict'):
        print('loading dictionary...')
        dictionary = corpora.Dictionary.load('tmp/dictionary.dict')       
    else:
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
        print('saving dictionary....')
        dictionary.save('tmp/dictionary.dict')
        # compute models
        print('computing BOW...')
        bow = Bow(inputFile, dictionary)
        print('computing TFIDF...')
        model = models.TfidfModel(bow)
        print('saving tfidf model...')
        model.save('tmp/tfidf_model')

    return dictionary


def computeDoc2Vec(inputFile):
    print('computing doc2vec...')
    documents = prepareTaggedDoc(inputFile)
    model = Doc2Vec(documents, vector_size=300)
    print('saving doc2vec model...')
    model.save('/tmp/doc2vec')


def extract_doc_rep(docFile, dictionary, rep='bow', name=''):
    if os.path.exists(os.path.join('tmp/', name + '.mm')):
        bow = corpora.MmCorpus(os.path.join('tmp/', name + '.mm'))
    else:
        bow = Bow(docFile, dictionary)
        corpora.MmCorpus.serialize(os.path.join('tmp/', name + '.mm'), bow)

    if rep == 'bow':     
        return corpus2csc(bow)
    elif rep == 'tfidf':
        print('loading tfidf model...')
        model = SaveLoad.load('tmp/tfidf_model')
        return corpus2csc(model[bow])
    elif rep == 'doc2vec':
        print('loading doc2vec model...')
        model = Doc2Vec.load('/tmp/doc2vec')
        
        
    else:
        print('supported representation: bow, tfidf, doc2vec')
        






