import os
import numpy as np
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import SaveLoad, simple_preprocess
from lxml.etree import iterparse
from utils import textCleaning

class IterCorpus():
    def __init__(self, file):
        self.file = file
 
    def __iter__(self):
        for event, elem in iterparse(self.file):
            if elem.tag == "article":
                articleId = elem.attrib['id']
                title = elem.attrib['title']
                text = "".join(elem.itertext())
                text = textCleaning(title, text)             
                yield(articleId, text)
                
                
class TaggedDoc(object):
    '''
    prepare tagged documents for the doc2vec model
    '''
    def __init__(self, file, word_num=1000):
        if isinstance(file, str):
            self.file = [file]
        else:
            self.file = file
        self.word_num = word_num

    def __iter__(self):
        for f in self.file:
            if 'byarticle' in f:
                fileIdx = 2
            elif 'validation' in f:
                fileIdx = 1
            else:
                fileIdx = 0
            corpus = IterCorpus(f)
            for text in corpus:
                ind = text[0] + '_' + str(fileIdx)
                text = ' '.join(text[1].split()[:self.word_num])
                yield TaggedDocument(simple_preprocess(text), [ind])


def getModelName(params):
    return '_'.join(['doc2vec', str(params['dim']), str(params['word_num'])])

def computeDoc2Vec(inputFileList, file_path, params):
    '''Build doc2vec model'''
    if os.path.exists(file_path + getModelName(params) ):
        return
    else:
        print('computing doc2vec...')
        train_corpus = TaggedDoc(inputFileList, params['word_num'])
        # training
        model = Doc2Vec(train_corpus, vector_size=params['dim'], min_count=10, epochs=20)
        print('saving doc2vec model...')
        model.save(file_path + getModelName(params))

def buildRep(dataPath, params):
    '''
    Build document representation model
    '''
    file_path = 'tmp/'
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    trainFile = dataPath + "articles-training-bypublisher.xml"
    valFile = dataPath + "articles-validation-bypublisher.xml"

    fileList = [trainFile, valFile]
    computeDoc2Vec(fileList, file_path, params)
    

if __name__ == "__main__":
    params = {
        'dim': 400,
        'word_num': 1000
        }
    buildRep("data/", params)






