import os
import numpy as np
from pathlib import Path
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import SaveLoad, simple_preprocess
from gensim.matutils import corpus2csc
from gensim.parsing.preprocessing import stem_text, remove_stopwords, strip_multiple_whitespaces
from gensim.parsing.preprocessing import preprocess_string, strip_non_alphanum, strip_numeric, strip_short

class TaggedDoc(object):
    '''
    prepare tagged documents for training the doc2vec model
    '''
    def __init__(self, fileList, sample_doc_num = 1000000, word_num = 1000):
        self.file = fileList
        self.sample_doc_num = sample_doc_num
        self.word_num = word_num
        remainIds = np.load('remain.npy')
        self.ids = remainIds

    def __iter__(self):
        if isinstance(self.file, str):
            self.file = [self.file]
        for f in self.file:
            for line_id, line in enumerate(open(f, encoding='utf-8')):
                if line_id in self.ids:
                    #raise StopIteration()
                    docId = line.split('::')[0]
                    title = line.split('::')[1]
                    text = line.split('::')[2].split()[:self.word_num]
                    text = title + ' ' + ' '.join(text)
                    yield TaggedDocument(simple_preprocess(text), [docId])

def getModelName(params):
    return '_'.join(['doc2vec', str(params['dim']), str(params['word_num'])])

def computeDoc2Vec(inputFileList, file_path, params):
    '''Build doc2vec model'''
    dim = params['dim']
    if os.path.exists(file_path + getModelName(params) ):
        return
    else:
        print('computing doc2vec...')
        train_corpus = TaggedDoc(inputFileList, params['doc_num'], params['word_num'])
        # training
        model = Doc2Vec(train_corpus, vector_size=dim, min_count=1, epochs=20)
        print('saving doc2vec model...')
        model.save(file_path +  getModelName(params))

def preprocess(text, stem):
    '''
    return a list of tokenized words filtered by filters
    '''
    filters = [remove_stopwords, strip_non_alphanum, strip_multiple_whitespaces, strip_numeric, strip_short]
    return preprocess_string(stem_text(text), filters) if stem else preprocess_string(text, filters)

def buildRep(inputFileFolder, params):
    '''
    Build document representation model

    InputFileFolder: path of the input file folder
    rep: representation (bow, tfidf or doc2vec)
    dim: dimension of the representation
    '''
    file_path = 'tmp/'
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    fileList = [inputFileFolder + f for f in os.listdir(inputFileFolder) if f.startswith('articles-') and f.endswith(".txt")]
    computeDoc2Vec(fileList, file_path, params)


def extract_doc_rep(textFile, params):
    '''
    Extract the representation for file docFile
    '''
    remainIds = np.load('remain.npy')
    file_path = 'tmp/'
    numDoc = 0
    if remainIds:
        numDoc = len(remainIds)
    else:
        for line in open(textFile, encoding='utf8'):
            numDoc = numDoc + 1

    print('loading doc2vec model...')
    model = Doc2Vec.load(file_path + getModelName(params) )
    # infer vectors
    vectors = np.zeros((params['dim'], numDoc))
    cnt = 0
    corpus = TaggedDoc(textFile, params['doc_num'], params['word_num'])
    for doc in corpus:
        if doc.tags[0] in model.docvecs.doctags.keys():
            vectors[:, cnt] = model[doc.tags[0]]
        else:
            vectors[:, cnt] = model.infer_vector(doc.words)
        cnt = cnt+1
    return vectors
        






