#!/usr/bin/env python
import re
import xml.sax
import lxml.sax
import lxml.etree
from nltk import word_tokenize, pos_tag
from lexicon_featurizer import *




groundTruth = {}
class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            hyperpartisan = attrs.getValue("hyperpartisan")
            groundTruth[articleId] = hyperpartisan

def handleArticleNLP(article, outFile):
    title = article.get('title')
    # get text from article
    text = lxml.etree.tostring(article, method="text").decode('utf-8')
    text_cleaned = re.sub('^[a-z]', '', text.lower())
    tokens = word_tokenize(text)
    tokens_cleaned = word_tokenize(text_cleaned)
    pos_tags = pos_tag(tokens)
    
    text_feats = extract_text_features(text, tokens_cleaned)    
    pos_feats = extract_pos_features(pos_tags)
    bias_feat = bias_lexicon(tokens_cleaned)
    sub_feat = subjective_lexicon(pos_tags)
    features = [text_feats, pos_feats, bias_feat, sub_feat]
    flattened = [val for sublist in features for val in sublist]

    outFile.write(article.get("id"))
    for f in flattened:
        outFile.write(" " + str(f))
    outFile.write(' ' + groundTruth[article.get("id")])
    outFile.write("\n")   


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
    with open(outputFile, 'w') as outFile:
        with open(trainFile) as inputFile:
            xml.sax.parse(inputFile, Featurizer(outFile))
