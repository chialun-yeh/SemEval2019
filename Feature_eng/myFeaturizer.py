#!/usr/bin/env python
import re
import xml.sax
import lxml.sax
import lxml.etree
import string
import nltk
import numpy as np
from text_featurizer import bias_lexicon, sentiment, subjective_lexicon, find_NE, extract_pos_features, extract_text_features


def extract_title_features(title):
    # CNN to process title
    # Use sentiment analysis for the title (extract NE, see the sentiment of the NE on Wiki)
    title_tokens = nltk.word_tokenize(title)
    title_words = [t.lower() for t in title_tokens if t not in string.punctuation]
    if len(title_words ) == 0:
        title_sent = [0,0,0,0,0,0]
        title_bias = [0]
        title_ner = 0
    else:
        title_sent = sentiment(title_words)
        title_bias = bias_lexicon(title_words)
        title_ner = len(find_NE(title))
    title_feat = [title_sent, title_bias, [title_ner]]
    return [val for sublist in title_feat for val in sublist]

def parseFeatures(filename, filepath = 'features/'):
    ids = []
    feats = []
    with open(filepath + filename, 'r', encoding='utf8') as feat:
        lines = feat.readlines()
        for line in lines:
            tmp = line.split()
            ids.append(tmp[0])
            feats.append(tmp[1:])
    X = np.asarray(feats).astype(float)
    return ids, X


def handleArticleNLP(article, outFile, use_features):
    title = article.get('title')
    title_feat = extract_title_features(title)
    # get text from article
    if use_features:
        text = lxml.etree.tostring(article, method="text", encoding='unicode')
        sentences = nltk.sent_tokenize(text)

        pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
        tokens = nltk.word_tokenize(text)
        words = [t.lower() for t in tokens if t not in string.punctuation]
    
        quotation = len(re.findall('"', text))
        text_feats = extract_text_features(sentences, words)    
        pos_feats = extract_pos_features(pos_tags)
        bias_feat = bias_lexicon(words)
        sub_feat = subjective_lexicon(pos_tags)
        sent_feats = sentiment(words)
        ner_feats = [len(find_NE(text))]
        features = [text_feats, pos_feats, bias_feat, sub_feat, sent_feats, ner_feats, [quotation]]
        flattened = [val for sublist in features for val in sublist]

    outFile.write(article.get("id"))
    for t in title_feat:
        outFile.write(' ' + str(t))
    if use_features:
        for f in flattened:
            outFile.write(' ' + str(f))
    outFile.write("\n")   


class Featurizer(xml.sax.ContentHandler):
    def __init__(self, outFile, use_features):
        xml.sax.ContentHandler.__init__(self)
        self.lxmlhandler = "undefined"
        self.outFile = outFile
        self.cnt = 0
        self.use_features = use_features

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
                if self.cnt % 1 == 0:
                    handleArticleNLP(self.lxmlhandler.etree.getroot(), self.outFile, self.use_features)
                self.lxmlhandler = "undefined"
                self.cnt = self.cnt+1         
