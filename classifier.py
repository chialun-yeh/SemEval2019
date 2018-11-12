import numpy as np
import os
import xml.sax
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from myFeaturizer import Featurizer



def parse(features):
    feats = []
    Y = []
    with open(features) as feat:
        lines = feat.readlines()
        for line in lines:
            tmp = line.split()
            feats.append(tmp[1:-1])
            Y.append(tmp[-1]) 
    X = np.asarray(feats).astype(float)
    return X, Y

if __name__ == '__main__':

    X, Y = parse('trn_feat.txt')
    clf = GaussianNB()
    scores = cross_val_score(clf, X, Y, cv=10)
    print('NB: ', np.mean(scores))
    clf = LogisticRegression()
    scores = cross_val_score(clf, X, Y, cv=10)
    print('LR: ', np.mean(scores))
