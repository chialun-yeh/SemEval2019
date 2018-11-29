import os
import pickle
import xml.sax
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from myFeaturizer import Featurizer
from doc_representation import *
from scipy.sparse import hstack


groundTruth = {}
class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            hyperpartisan = attrs.getValue("hyperpartisan")
            groundTruth[articleId] = hyperpartisan

def parseFeatures(filename, filepath = './features'):
    ids = []
    feats = []
    with open(os.path.join(filepath, filename)) as feat:
        lines = feat.readlines()
        for line in lines:
            tmp = line.split()
            ids.append(tmp[0])
            feats.append(tmp[1:])
    X = np.asarray(feats).astype(float)
    return ids, X

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
    test = True
    if test:
        name = 'sample'
        trainFile = 'data/sampleArticle_trn.xml'
        labelFile = 'data/ground_trn.xml'
        allDocs = ['features/test_doc.txt']
    else:
        name = 'trn'
        trainFile = 'data/articles-training-bypublisher.xml'
        labelFile = 'data/ground-truth-training-bypublisher.xml'
        allDocs = ['features/train_doc.txt', 'features/val_doc.txt']

    # parse groundTruth
    with open(labelFile) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())

    # extract doc representation 
    dic = createDict(allDocs)
    X_rep = extract_doc_rep(allDocs[0], dic, 'bow', name='trn_rep')


    if use_features:
        feature_path = './features/'
        # extract and write features to ./features/
        with open(os.path.join(feature_path, name) + '.txt', 'w') as outFile:
            with open(trainFile, encoding='utf-8') as inputFile:
                parser = xml.sax.make_parser()
                parser.setContentHandler(Featurizer(outFile))
                source = xml.sax.xmlreader.InputSource()
                source.setByteStream(inputFile)
                source.setEncoding('utf-8')
                parser.parse(source)
        # parse features
        ids, feats = parseFeatures(name + '.txt')
        X_trn = hstack(X_rep.transpose(), feats)

    else:
        X_trn = X_rep.transpose()
        ids = [line.split(',')[0] for line in open(allDocs[0])]
    
    # build label for training
    Y_trn = []
    for id in ids:
        Y_trn.append(groundTruth[id])

    # train model
    train_model(X_trn, Y_trn, 'lr')

    

