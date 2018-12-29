"""
Adapted from https://github.com/1991viet/Hierarchical-attention-networks-pytorch
@author: Viet Nguyen
@author: chialunyeh
"""

import csv
import xml
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize

groundTruth = {}
class GroundTruthHandler(xml.sax.ContentHandler):
    '''
    Read labels from xml and save in groudTruth as groundTruth[article_id] = label
    '''
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            hyperpartisan = attrs.getValue("hyperpartisan")
            groundTruth[articleId] = hyperpartisan

class MyDataset(Dataset):
    def __init__(self, data_path, dict_path, label_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        if label_path is not '':
            with open(label_path) as groundTruthDataFile:
                xml.sax.parse(groundTruthDataFile, GroundTruthHandler())

        articleIds, texts, labels = [], [], []
        with open(data_path, encoding = 'utf8') as file:
            for line in file:
                articleId = line.split('::')[0]
                title = line.split('::')[1]
                text = line.split('::')[2]
                texts.append(title + text)
                if label_path is not '':
                    if groundTruth[articleId] == 'true':
                        labels.append(int(1))
                    else:
                        labels.append(int(0))
                else:
                    labels.append(int(0))
                articleIds.append(articleId)

        self.articleIds = articleIds
        self.texts = texts
        self.labels = labels
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        articleId = self.articleIds[index]

        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(sentences)] \
            for sentences in sent_tokenize(text)]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][:self.max_length_sentences]
        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return articleId, document_encode.astype(np.int64), label

if __name__ == '__main__':
    test = MyDataset("../data/articles-training-byarticle.txt", "../pre-trained/glove.6B.100d.txt", '')
    print (test.__getitem__(index=1)[0].shape)