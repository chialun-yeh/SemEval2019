#!/usr/bin/env python
import re
import xml.sax
import lxml.sax
import lxml.etree
from text_featurizer import *

def processTitle(title):
    # CNN to process title
    # Use sentiment analysis for the title (extract NE, see the sentiment of the NE on Wiki)
    named_entities = find_NE(title)
    ne_num = len(named_entities)
    return [ne_num]

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


def handleArticleNLP(article, outFile):
    title = article.get('title')
    title_tokens = nltk.word_tokenize(title)
    title_words = [t.lower() for t in title_tokens if t not in string.punctuation]
    if len(title_words ) == 0:
        print(article.get('id'))
        title_sent = [0,0,0,0,0,0]
        title_bias = 0
        title_ner = 0
    else:
        title_sent = sentiment(title_words)
        title_bias = bias_lexicon(title_words)
        title_ner = len(find_NE(title))
    title_feats = [title_bias, title_ner]

    # get text from article
    text = lxml.etree.tostring(article, method="text", encoding='unicode')
    sentences = nltk.sent_tokenize(text)

    pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
    tokens = nltk.word_tokenize(text)
    words = [t.lower() for t in tokens if t not in string.punctuation]
    
    quotation = len(re.findall('"', text))
    title_feats = processTitle(title)
    text_feats = extract_text_features(sentences, words)    
    pos_feats = extract_pos_features(pos_tags)
    bias_feat = bias_lexicon(words)
    sub_feat = subjective_lexicon(pos_tags)
    sent_feats = sentiment(words)
    ner_feats = [len(find_NE(text))]
    features = [title_feats, title_sent, text_feats, pos_feats, bias_feat, sub_feat, sent_feats, ner_feats, [quotation]]
    flattened = [val for sublist in features for val in sublist]

    outFile.write(article.get("id") + ' ')
    for f in flattened:
        outFile.write(' ' + str(f))
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
                if self.cnt % 1 == 0:
                    handleArticleNLP(self.lxmlhandler.etree.getroot(), self.outFile)
                self.lxmlhandler = "undefined"
                self.cnt = self.cnt+1         
