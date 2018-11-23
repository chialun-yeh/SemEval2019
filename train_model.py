import os
import xml.sax
import numpy as np
from sklearn.linear_model import LogisticRegression
from myFeaturizer import Featurizer
import pickle



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

if __name__ == '__main__':
    name = 'sampleTrn'
    data_path = './data/articles-training-.xml/'
    feature_path = './features/'
    label_path = './data/ground-truth-training-20180831.xml/'

    with open(os.path.join(feature_path, name) + '.txt', 'w') as outFile:
        with open(os.path.join(data_path, name) + '.xml') as inputFile:
            xml.sax.parse(inputFile, Featurizer(outFile))

    # parse groundTruth
    with open(os.path.join(label_path, name) + '.xml') as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())

    # parse features
    ids, X = parseFeatures('trn_feat.txt')
    Y = []
    for id in ids:
        Y.append(groundTruth[id])

    # train model
    model = LogisticRegression()
    model.fit(X, Y)
    model_path = './models/'
    model_name = 'lr.sav'
    pickle.dump(model, open(os.path.join(model_path, model_name),'wb'))

