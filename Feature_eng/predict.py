#!/usr/bin/env python

"""Random baseline for the PAN19 hyperpartisan news detection task"""
# Version: 2018-09-24

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputDir=<directory>
#   Directory to which the predictions will be written. Will be created if it does not exist.

from __future__ import division

import os
import sys
import getopt
import pickle
import xml.sax
import lxml.sax
import lxml.etree
import numpy as np
from scipy import sparse
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, simple_preprocess
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from doc_representation import preprocess, buildName
from myFeaturizer import extract_title_features
from handleXML import cleanText, cleanQuotations

# global parameters
runOutputFileName = "prediction.txt"
stem = True
rep = 'bow'
dim = 50000
use_title = False
use_features = False
model = pickle.load(open('model/bow_stem_lr.sav', 'rb'))

rep_file = 'tmp/'
if rep == 'bow' or rep =='tfidf':
    dct = corpora.Dictionary.load(rep_file + buildName(stem, 'dictionary') + '.dict')
if rep == 'tfidf':
    rep_model = SaveLoad.load(rep_file + buildName(stem, 'tfidf_model'))
elif rep == 'doc2vec':
    rep_model = Doc2Vec.load( rep_file + 'doc2vec')

def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)


def handleArticle(article, outFile):
    # get text from article
    title = article.get('title')
    title = cleanQuotations(title)
    text = lxml.etree.tostring(article, method="text", encoding="unicode")
    text = cleanQuotations(text)
    text = cleanText(text)
    text = title + ' ' + text
    if rep == 'bow' or rep == 'tfidf':
        bow = dct.doc2bow(preprocess(text, stem))
        if rep == 'bow':
            X = corpus2csc([bow], num_terms=dim)
        else:
            X = corpus2csc([rep_model[bow]], num_terms=dim)
    else:
        X = rep_model.infer_vector(simple_preprocess(text))
    
    if use_title:
        title = article.get('title')
        title_feat = extract_title_features(title)
        if rep == 'doc2vec':
            X = np.hstack((X, title_feat))
        else:
            X = sparse.hstack((X.T, title_feat))

    articleId = article.get("id")
    prediction = model.predict(X.T)
   # prediction = model.predict(X.reshape(1, -1))
    outFile.write(articleId + " " + prediction[0] + "\n")

class Predictor(xml.sax.ContentHandler):
    def __init__(self, outFile):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile
        self.lxmlhandler = "undefined"

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
                handleArticle(self.lxmlhandler.etree.getroot(), self.outFile)                
                self.lxmlhandler = "undefined"


########## MAIN #########
def main(inputDataset, outputDir):
    """Main method of this module."""

    with open(outputDir + "/" + runOutputFileName, 'w', encoding = 'utf8') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                with open(inputDataset + "/" + file, 'r', encoding = 'utf8') as inputRunFile:
                    parser = xml.sax.make_parser()
                    parser.setContentHandler(Predictor(outFile))
                    source = xml.sax.xmlreader.InputSource()
                    source.setByteStream(inputRunFile)
                    parser.parse(source)


if __name__ == '__main__':
    #main(*parse_options())
    main('../data/article', 'predictions')