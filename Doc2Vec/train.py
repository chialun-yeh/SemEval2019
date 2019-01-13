import os
import pickle
import xml.sax
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import numpy as np
from doc_representation import buildRep, extract_doc_rep, getModelName
from handleXML import createDocuments

groundTruth = {}
params = {
        'dim': 400,
        'doc_num': 100000,
        'word_num': 300
    }
clsf = 'rf'
remainIds = np.load('remain.npy')

class GroundTruthHandler(xml.sax.ContentHandler):
    '''
    Read labels from xml and save in groudTruth as groundTruth[article_id] = label
    '''
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            hyperpartisan = attrs.getValue("hyperpartisan")
            groundTruth[articleId] = hyperpartisan

def train_model(X, Y, clsf, model_name, outputPath):
    """Train main model with input features X and label Y."""
    if clsf == 'lr':
        clsf = LogisticRegression(solver='lbfgs')

    elif clsf == 'rf':
        clsf = RandomForestClassifier()
  
    print('training model')
    clsf.fit(X, Y)
    pred = clsf.predict(X)
    print('Training accuracy: ', accuracy_score(Y, pred))
    pickle.dump(clsf, open(outputPath + '/' + model_name,'wb'))

def main(inputFile, labelFile, outputPath, idPath):  
    
    """Main method of this module."""

    ''' 
    Pre-processing for word-based method: lower everything, strip all punctuation, lemmatize/stemming
    Word-based representation: BOW, TFIDF, ngrams 
    Learned represenatation: doc2vec

    Tunable: 
    [I] document representation
        3. doc2vec: vector_size, window, min_count
    [III] models
        1. algorithm: LR, RF, NB, SVM
        2. parameters of the algorithm
    '''

    modelName = getModelName(params) + '_' + clsf+ '.sav'

    # parse groundTruth
    with open(labelFile) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())
    
    # build document representation model
    text_folder = '../data/'
    text_name = Path(inputFile).name.strip('.xml') + '.txt'
    buildRep(text_folder, params)
    # extract doc representation for training set
    X_rep = extract_doc_rep(text_folder + text_name, params)
    X_trn = X_rep.transpose()
    articleIds = [line.split('::')[0] for line_id, line in enumerate(open(text_folder + text_name, encoding='utf8')) if line_id in remainIds]
    
    X_trn = normalize(X_trn)
    # build label for training
    Y_trn = []
    for articleId in articleIds:
        Y_trn.append(groundTruth[articleId])

    # train model
    train_model(X_trn, Y_trn, clsf, modelName, outputPath)


if __name__ == '__main__':
    inputFile = '../data/articles-training-bypublisher.xml'
    labelFile = '../data/ground-truth-training-bypublisher.xml'
    outputPath = 'model'
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    main(inputFile, labelFile, outputPath, 'remain.npy')