#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import xml.sax
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from myFeaturizer import Featurizer, parseFeatures
from doc_representation import buildRep, extract_doc_rep
from scipy.sparse import hstack


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

def train_model(X, Y, model):
    model_path = './models/'
    if model == 'lr':
        model = LogisticRegression()
        model_name = 'logisticRegression.sav'

    elif model == 'rf':
        model = RandomForestClassifier()
        model_name = 'randomForest.sav'
    print('training model')
    model.fit(X, Y)
    pickle.dump(model, open(os.path.join(model_path, model_name),'wb'))

if __name__ == '__main__':
    use_features = False
    test = False
    if test:
        run_name = 'sample'
        trainFile = 'sample_data/articles-training-bypublisher.xml'
        labelFile = 'data/ground-truth-training-bypublisher.xml'
        doc4Dict = ['features/sample-trn.txt', 'features/sample-val.txt']
    else:
        run_name = 'trn'
        trainFile = 'data/articles-training-bypublisher.xml'
        labelFile = 'data/ground-truth-training-bypublisher.xml'
        doc4Dict = ['features/train_doc.txt', 'features/val_doc.txt']

    # parse groundTruth
    with open(labelFile) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())

    # build representation model
    rep = 'tfidf'
    dim = 50000
    buildRep(doc4Dict, rep, dim, test)
    # extract doc representation 
    X_rep = extract_doc_rep(doc4Dict[0], rep, name='trn_rep', sample=test)

    if use_features:
        feature_path = './features/'
        # extract and write features to ./features/
        with open(os.path.join(feature_path, run_name) + '.txt', 'w') as outFile:
            with open(trainFile, encoding='utf-8') as inputFile:
                parser = xml.sax.make_parser()
                parser.setContentHandler(Featurizer(outFile))
                source = xml.sax.xmlreader.InputSource()
                source.setByteStream(inputFile)
                source.setEncoding('utf-8')
                parser.parse(source)
        # parse features
        ids, feats = parseFeatures(run_name + '.txt')
        X_trn = hstack(X_rep.transpose(), feats)

    else:
        X_trn = X_rep.transpose()
        ids = [line.split(',')[0] for line in open(doc4Dict[0])]
    
    # build label for training
    Y_trn = []
    for i in ids:
        Y_trn.append(groundTruth[i])

    # train model
    train_model(X_trn, Y_trn, 'lr')

    

