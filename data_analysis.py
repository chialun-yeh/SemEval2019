#!/usr/bin/env python

import xml.sax
import lxml.sax
import lxml.etree
import re
import nltk
from nltk import word_tokenize, pos_tag
import numpy as np
from lexicon_featurizer import *
import matplotlib.pyplot as plt

def switcher(pos):
    pos_family = {
    'NN': 'noun',
    'NNS': 'noun',
    'NNP': 'noun',
    'NNPS': 'noun',
    'VB': 'verb',
    'VBD': 'verb',
    'VBG': 'verb',
    'VBN': 'verb',
    'VBP': 'verb',
    'VBZ': 'verb',
    'JJ': 'adj',
    'JJR': 'adj',
    'JJS': 'adj',
    'PRP': 'pron',
    'PRP$': 'pron',
    'WP': 'pron',
    'WP$': 'pron',
    'RB': 'adverb',
    'RBR': 'adverb',
    'RBS': 'adverb'}

    if pos in pos_family.keys():
        return pos_family[pos]
    else:
        return 'others'

groundTruth = {}
class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            hyperpartisan = attrs.getValue("hyperpartisan")
            groundTruth[articleId] = hyperpartisan


sub_true = []
sub_false = []
bias_true = []
bias_false = []
adverb_true = []
adverb_false = []

def handleArticleNLP(article, outFile):
    title = article.get('title')
    # get text from article
    text = lxml.etree.tostring(article, method="text").decode('utf-8')
    text_cleaned = re.sub('^[a-z]', '', text.lower())
    tokens = word_tokenize(text)
    tokens_cleaned = word_tokenize(text_cleaned)
    
    # pos
    pos_tags = pos_tag(tokens)
    fixed_tags = []
    pos_dict = {'noun':0, 'verb':0, 'adj':0, 'pron':0, 'adverb' :0, 'others':0}
    for t in pos_tags:
        pos = switcher(t[1])
        fixed_tags.append((t[0], pos))
        pos_dict[pos] = pos_dict[pos] + 1

    bias = bias_lexicon(tokens_cleaned)
    sub = subjective_lexicon(fixed_tags)
    if groundTruth[article.get("id")] == 'true':
        bias_true.append(bias)
        sub_true.append(sub)
        adverb_true.append(pos_dict['adverb'])
    else:
        bias_false.append(bias)
        sub_false.append(sub)
        adverb_false.append(pos_dict['adverb'])

class Featurizer(xml.sax.ContentHandler):
    def __init__(self, outFile):
        xml.sax.ContentHandler.__init__(self)
        self.lxmlhandler = "undefined"
        self.outFile = outFile
        self.cnt = 0

    def startElement(self, name, attrs):
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()
            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                # pass to handleArticle function
                if self.cnt % 500 == 0:
                    handleArticleNLP(self.lxmlhandler.etree.getroot(), self.outFile)
                self.lxmlhandler = "undefined"
                self.cnt = self.cnt+1
                

                           
if __name__ == '__main__':

    # Parse groundTruth
    train_label = "C:/Users/sharo/Documents/SemEval2019/data/ground-truth-training-20180831.xml/yTrn.xml"
    with open(train_label) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())


    trainFile = "C:/Users/sharo/Documents/SemEval2019/data/articles-training-20180831.xml/xTrn.xml"
    outputFile = 'trn_feat.txt'
    test = 'sampleData.xml'
    with open(outputFile, 'w') as outFile:
        with open(trainFile) as inputFile:
            xml.sax.parse(inputFile, Featurizer(outFile))

    print(np.mean(bias_true), np.mean(bias_false))
    print(np.mean(sub_true), np.mean(sub_false))
    print(np.mean(adverb_true), np.mean(adverb_false))


    fig, ax = plt.subplots()
    ax.boxplot([bias_true, bias_false])
    fig1, ax1 = plt.subplots()
    ax1.boxplot([sub_true, sub_false])
    fig2, ax2 = plt.subplots()
    ax2.boxplot([adverb_true, adverb_false])
    plt.show()


