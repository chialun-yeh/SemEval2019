# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 12:34:03 2018
@author: chialunyeh
"""

import xml.etree.ElementTree as ET
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def readFile(file):
    with open(file, encoding='utf-8') as f:
        for line in f:
            yield line.split('::')[1]
            

def readLabel(file):
    y = []
    with open(file, encoding="utf-8") as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for article in root.iter('article'):
            y.append(article.attrib['hyperpartisan'])
            
    return y
                

train_file = '../data/articles-training-bypublisher.txt'
dev_file = '../data/articles-validation-bypublisher.txt'
test_file = '../data/articles-training-byarticle.txt'

corpus_train = readFile(train_file)
corpus_dev = readFile(dev_file)
corpus_test = readFile(test_file)
label_train = readLabel('../data/ground-truth-training-bypublisher.xml')
label_dev = readLabel('../data/ground-truth-validation-bypublisher.xml')
label_test = readLabel('../data/ground-truth-training-byarticle.xml')


vectorizer = HashingVectorizer(ngram_range=(1, 3))
model = LogisticRegression()
X = vectorizer.fit_transform(corpus_train)
#pos = nltk.pos_tag(nltk.word_tokenize(title))
model.fit(X, label_train)

dev = vectorizer.transform(corpus_dev)
tst = vectorizer.transform(corpus_test)
dev_pred = model.predict(dev)
tst_pred = model.predict(tst)
print('Dev accuracy: %f.4' %accuracy_score(label_dev, dev_pred))
print('Test accuracy: ', accuracy_score(label_test, tst_pred))
#confusion_matrix(label_test, tst_pred)