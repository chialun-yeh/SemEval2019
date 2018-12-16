import os
import sys
import re
import getopt
import pickle
import xml.sax
import lxml.sax
import lxml.etree
from pathlib import Path
from myFeaturizer import Featurizer, extract_title_features
from train import parseFeatures
from scipy.sparse import hstack
from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad
from gensim import corpora, models



runOutputFileName = "prediction.txt"
dim = 50000
model = pickle.load(open('model/tfidf_title_lr.sav', 'rb'))
tfidf_model = SaveLoad.load('tmp/tfidf_model')


########## SAX ##########
def handleArticle(article, outFile):
    # get text from article
    title = article.get('title')
    title_feat = extract_title_features(title)

    text = lxml.etree.tostring(article, method="text", encoding="unicode")
    text = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", text)
    text = re.sub(r'[„“]|(\'\')|(,,)', '"', text)
    text = re.sub(r'(www\S+)|(http\S+)|(href)|(\S+.com)', '', text)
    text = re.sub(r'\{.+\} | \[.+\]', '', text)
    text = text.strip('Getty Images').strip('Getty')
    text = re.sub(r'@\S+ | #\S+', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.lstrip().replace('\n','')
    dictionary = corpora.Dictionary.load('tmp/dictionary.dict')  
    bow = dictionary.doc2bow(text.split())
    X = corpus2csc([tfidf_model[bow]], num_terms=dim)
    X_val = hstack( (X.T, title_feat))
    articleId = article.get("id")
    prediction = model.predict(X_val)
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
    main('data/article', 'predictions')
