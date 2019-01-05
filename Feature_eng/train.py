import sys
import os
import getopt
import argparse
import pickle
import xml.sax
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from scipy import sparse
import numpy as np

from myFeaturizer import Featurizer, parseFeatures
from doc_representation import buildRep, extract_doc_rep
from handleXML import createDocuments


groundTruth = {}
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

def train_model(X, Y, model, outputPath):
    """Train main model with input features X and label Y."""
    if model == 'lr':
        model = LogisticRegression()
        model_name = 'logisticRegression.sav'

    elif model == 'rf':
        model = RandomForestClassifier()
        model_name = 'randomForest.sav'
    print('training model')
    model.fit(X, Y)
    pickle.dump(model, open(outputPath + '/' + model_name,'wb'))

def main(inputFile, labelFile, outputPath):  
    
    """Main method of this module."""

    ''' 
    Pre-processing for word-based method: lower everything, strip all punctuation, lemmatize/stemming
    Word-based representation: BOW, TFIDF, ngrams 
    Learned represenatation: doc2vec

    Tunable: 
    [I] document representation
        1. stem or not
        2. bow/tfidf: vector_size
        3. doc2vec: vector_size, window, min_count
    [II] additional features
        1. title: 
        2. text
    [III] models
        1. algorithm: LR, RF, NB, SVM
        2. parameters of the algorithm
    '''

    stem = True
    rep = 'tfidf'
    dim = 50000
    use_title = False
    use_features = False
    model = 'lr'

    # parse groundTruth
    with open(labelFile) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())
    
    # build document representation model
    text_folder = '../data/'
    if not os.path.exists(text_folder):
        os.mkdir(text_folder)
    text_name = Path(inputFile).name.strip('.xml') + '.txt'
    if not os.path.exists(text_folder + text_name ):
        print('creating text document...')
        createDocuments(inputFile, text_folder + text_name)
    else:
        print('loading text document...')
    
    buildRep(text_folder, rep, dim, stem)
    # extract doc representation for training set
    X_rep = extract_doc_rep(text_folder + text_name, rep, dim, stem)

    if use_features or use_title:
        feature_path = 'features/'
        if not os.path.exists(feature_path):
            os.mkdir(feature_path)
        if not os.path.exists(feature_path + text_name ):
            print('computing features....')
            # extract and write features to ./features/
            with open(feature_path + text_name, 'w') as outFile:
                with open(inputFile, 'r', encoding='utf-8') as inputFile:
                    parser = xml.sax.make_parser()
                    parser.setContentHandler(Featurizer(outFile, use_features))
                    source = xml.sax.xmlreader.InputSource()
                    source.setByteStream(inputFile)
                    parser.parse(source)

        # parse features
        articleIds, feats = parseFeatures(text_name)
        if rep == 'doc2vec':
            X_trn = np.hstack( (X_rep.transpose(), feats))
        else:
            X_trn = sparse.hstack( (X_rep.transpose(), feats))

    else:
        X_trn = X_rep.transpose()
        articleIds = [line.split('::')[0] for line in open(text_folder + text_name, encoding='utf8')]
    
    X_trn = normalize(X_trn)
    # build label for training
    Y_trn = []
    for articleId in articleIds:
        Y_trn.append(groundTruth[articleId])

    # train model
    train_model(X_trn, Y_trn, model, outputPath)


if __name__ == '__main__':
    inputFile = '../data/articles-training-bypublisher.xml'
    labelFile = '../data/ground-truth-training-bypublisher.xml'
    outputPath = 'model'
    main(inputFile, labelFile, outputPath)