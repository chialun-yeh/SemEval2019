#!/usr/bin/env python
import re
import xml.sax
import lxml.sax
import lxml.etree
import nltk
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

groundTruth = {}
class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            hyperpartisan = attrs.getValue("hyperpartisan")
            groundTruth[articleId] = hyperpartisan


subjectivity = [[], []]
bias = [[], []]
adverb = [[], []]
adj = [[], []]
title_length = [[], []]
word_count = [[], []]
word_length = [[], []]
sentence_count = [[], []]
link_count = [[], []]
para_count = [[], []]
quo_count = [[], []]


def handleArticleNLP(article, links, paragraphs):
    title = article.get('title')
    title_tokens = nltk.word_tokenize(title)
    title_words = [t.lower() for t in title_tokens if t not in string.punctuation]
    if len(title_words ) == 0:
        print(title)
        title_sent = 0
        title_bias = 0
        title_ner = 0
    else:
        title_sent = sentiment(title_words)
        title_bias = bias_lexicon(title_words)
        title_ner = len(find_NE(title))
    title_feats = [title_bias, title_ner]

    # get text from article
    text = lxml.etree.tostring(article, method="text").decode('utf-8')
    sentences = nltk.sent_tokenize(text)

    pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
    tokens = nltk.word_tokenize(text)
    words = [t.lower() for t in tokens if t not in string.punctuation]
    text_feats = extract_text_features(sentences, words)    
    bias_feat = bias_lexicon(words)
    sub_feat = subjective_lexicon(pos_tags)
    sent_feats = sentiment(words)
    ner_feats = [len(find_NE(text))]
    features = [title_feats, title_sent, text_feats, bias_feat, sub_feat, sent_feats, ner_feats]
    flattened = [val for sublist in features for val in sublist]

    quotation = len(re.findall('"', text))
    if groundTruth[article.get("id")] == 'true':
        a = 0
    else:
        a = 1
        '''
    subjectivity[a].append(sub)
    bias[a].append(bias_score)
    title_length[a].append(title_len)
    word_count[a].append(word_num)
    word_length[a].append(word_len)
    sentence_count[a].append(sent_num)
    adverb[a].append(pos_dict['adverb'])
    adj[a].append(pos_dict['adj'])
    link_count[a].append(links/word_num)
    para_count[a].append(paragraphs)
    quo_count[a].append(quotation)'''

class Featurizer(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
        self.lxmlhandler = "undefined"
        self.cnt = 0
        self.links = 0
        self.paragraphs = 0

    def startElement(self, name, attrs):
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()
            if name == 'p':
                self.paragraphs = self.paragraphs + 1
            if name == 'a':
                self.links = self.links + 1
            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                # pass to handleArticle function
                if self.cnt % 10 == 0:
                    handleArticleNLP(self.lxmlhandler.etree.getroot(), self.links, self.paragraphs)
                if self.cnt % 500 == 0:
                    print('Round: ', self.cnt)
                self.lxmlhandler = "undefined"
                self.cnt = self.cnt+1
                self.paragraphs = 0
                self.links = 0
                

                           
if __name__ == '__main__':
    # Parse groundTruth
    train_label = "C:/Users/sharo/Documents/SemEval2019/data/ground-truth-training-20180831.xml/yTrn.xml"
    val_label = "C:/Users/sharo/Documents/SemEval2019/data/ground-truth-validation-20180831.xml/yVal.xml"
    with open(val_label) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())

    trainFile = "C:/Users/sharo/Documents/SemEval2019/data/articles-training-20180831.xml/xTrn.xml"
    valFile = "C:/Users/sharo/Documents/SemEval2019/data/articles-validation-20180831.xml/xVal.xml"
    testFile = 'C:/Users/sharo/Documents/SemEval2019/data/test.xml'
    with open(testFile) as inputFile:
        xml.sax.parse(inputFile, Featurizer())

    #print(len(title_length[0]))
    #print(np.mean(title_length[0]), np.mean(title_length[1]))
    #print(np.mean(word_count[0]), np.mean(word_count[1]))
    #print(np.mean(word_length[0]), np.mean(word_length[1]))

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].boxplot(title_length)
    axs[0,0].set_title('Title length')
    axs[0, 1].boxplot(word_count)
    axs[0,1].set_title('word count')
    axs[0,2].boxplot(word_length)
    axs[0,2].set_title('word length')
    axs[1,0].boxplot(sentence_count)
    axs[1,0].set_title('sentence count')
    axs[1,1].boxplot(link_count)
    axs[1,1].set_title('link count')
    axs[1,2].boxplot(para_count)
    axs[1,2].set_title('paragraph count')

    fig1, ax1 = plt.subplots(2, 3)
    ax1[0,0].boxplot(bias)
    ax1[0,0].set_title('Bias')
    ax1[0,1].boxplot(subjectivity)
    ax1[0,1].set_title('Subjectivity')
    ax1[1,0].boxplot(adverb)
    ax1[1,0].set_title('Adverb')
    ax1[1,1].boxplot(adj)
    ax1[1,1].set_title('Adj')
    ax1[1,2].boxplot(quo_count)
    ax1[1,2].set_title('quotation counts')
    plt.show()
