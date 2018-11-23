#!/usr/bin/env python

"""Term frequency extractor for the PAN19 hyperpartisan news detection task"""
# Version: 2018-10-09

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputFile=<file>
#   File to which the term frequency vectors will be written. Will be overwritten if it exists.

# Output is one article per line:
# <article id> <token>:<count> <token>:<count> ...


import os
import getopt
import sys
import xml.sax
import lxml.sax
import lxml.etree
import re
from collections import Counter
from nltk.corpus import stopwords


termfrequencies = {}
########## ARTICLE HANDLING ##########
def handleArticle(article):
    # get text from article
    text = lxml.etree.tostring(article, method="text").decode('utf-8')
    textcleaned = re.sub('[^a-z ]', '', text.lower())
    tokens = [t for t in textcleaned.split() if t not in stopwords.words('english')]
    # counting tokens
    for token in tokens:
        if token in termfrequencies:
            termfrequencies[token] += 1
        else:
            termfrequencies[token] = 1


########## SAX FOR STREAM PARSING ##########
class TFExtractor(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
        self.lxmlhandler = "undefined"
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
                if self.cnt % 10 == 0:
                # pass to handleArticle function
                    handleArticle(self.lxmlhandler.etree.getroot())
                self.lxmlhandler = "undefined"
                self.cnt = self.cnt+1


    

if __name__ == '__main__':

    trainFile = "C:/Users/sharo/Documents/SemEval2019/data/articles-training-20180831.xml/xTrn.xml"
    outputFile = 'term_frequency.txt'
    with open(trainFile) as inputRunFile:
        xml.sax.parse(inputRunFile, TFExtractor())

    c = Counter(termfrequencies)
    with open(outputFile, 'w') as outFile:
        for token, count in c.most_common(50000):
            outFile.write(str(token) + ' ' + str(count) + '\n')

