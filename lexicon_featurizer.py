import os
import nltk
import numpy as np
from gensim.models import KeyedVectors

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


def bias_lexicon(tokens):
    total_count = 0
    with open ('lexicons/bias-lexicon/bias-lexicon.txt') as corpus:
        lexicon = corpus.read().split()
    for t in tokens:
        if t in lexicon:
            total_count = total_count +1
    return [total_count]

def subjective_lexicon(pos_tags):
    model = KeyedVectors.load_word2vec_format('lexicons//GoogleNews-vectors-negative300.bin.gz', binary=True)
    total_count = 0
    lexicon = {}
    with open ('lexicons/processed_subj.txt') as corpus:
        lines = corpus.readlines()
        for line in lines:
            fields = line.split()
            lexicon[fields[0]] = (fields[1], fields[2])
    for item in pos_tags:
        if item[0] in lexicon.keys():
            if lexicon[item[0]][0] == switcher(item[1]) or lexicon[item[0]][0] == 'anypos':
                total_count = total_count +1
    return [total_count]

def extract_text_features(text, tokens_cleaned):
    # word count
    word_num = len(tokens_cleaned)
    chars = [len(t) for t in tokens_cleaned]
    # character count
    char_num = np.sum(chars)
    # average length of word
    word_len = np.mean(chars)
    # punctuation count
    # upper case count
    sentence_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_tokenizer.tokenize(text)
    # sentence count
    sent_num = len(sentences)
    # sentence length
    sent_len = np.mean([len(s.split(' ')) for s in sentences])

    return [word_num, char_num, word_len, sent_num, sent_len]


def extract_pos_features(pos_tags):
    pos_dict = {'noun':0, 'verb':0, 'adj':0, 'pron':0, 'adverb':0, 'others':0}
    for t in pos_tags:
        pos = switcher(t[1])
        pos_dict[pos] = pos_dict[pos] + 1
    feat = []
    for k in pos_dict:
        if k != 'others':
            feat.append(pos_dict[k])
    return feat