import nltk
import numpy as np
import os

def find_NE(text = ''):
    nes = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                nes.append( (chunk.label(), ' '.join(c[0] for c in chunk))) 
    return nes

def extract_text_features(sentences, words):
    stopwords =  nltk.corpus.stopwords.words('english')

    # word count
    word_num = len(words)
    stopwords_count = float(sum([words.count(w) for w in stopwords]))/len(words)
    all_cap_num = len([w for w in words if w.isupper()])
    chars = [len(w) for w in words]
    # average length of word
    word_len = np.mean(chars)
    # sentence count
    sent_num = len(sentences)
    # sentence length
    sent_len = np.mean([len(s.split(' ')) for s in sentences])

    return [word_num, stopwords_count, all_cap_num, word_len, sent_num, sent_len]


def switcher(pos):
    pos_family = {'NN': 'noun', 'NNS': 'noun', 'NNP': 'noun', 'NNPS': 'noun',
    'VB': 'verb', 'VBD': 'verb', 'VBG': 'verb', 'VBN': 'verb', 'VBP': 'verb', 'VBZ': 'verb',
    'JJ': 'adj', 'JJR': 'adj', 'JJS': 'adj',
    'PRP': 'pron', 'PRP$': 'pron', 'WP': 'pron', 'WP$': 'pron',
    'RB': 'adverb', 'RBR': 'adverb', 'RBS': 'adverb'}

    if pos in pos_family.keys():
        return pos_family[pos]
    else:
        return 'others'

def load_bias_lexicon(filepath = 'lexicons/'):
    with open (os.path.join(filepath, 'bias-lexicon.txt')) as corpus:
        words = corpus.read().split()

    return words

def load_subj_lexicon(filepath = 'lexicons/'):
    subjective_words = {}
    with open (os.path.join(filepath, 'processed_subj.txt')) as corpus:
        lines = corpus.readlines()
        for line in lines:
            fields = line.split()
            subjective_words[fields[0]] = (fields[1], fields[2])
    return subjective_words

def load_sentiment(filepath = 'lexicons/'):
    with open(os.path.join(filepath, "subjclueslen.txt")) as lex:
        wneg = set([])
        wpos = set([])
        wneu = set([])
        sneg = set([])
        spos = set([])
        sneu = set([])
        for l in lex.readlines():
            line = l.split()
            if line[0] == "type=weaksubj":
                if line[-1] == "priorpolarity=negative":
                    wneg.add(line[2].split("=")[1])
                elif line[-1] == "priorpolarity=positive":
                    wpos.add(line[2].split("=")[1])
                elif line[-1] == "priorpolarity=neutral":
                    wneu.add(line[2].split("=")[1])
                elif line[-1] == "priorpolarity=both":
                    wneg.add(line[2].split("=")[1])
                    wpos.add(line[2].split("=")[1])
            elif line[0] == "type=strongsubj":
                if line[-1] == "priorpolarity=negative":
                    sneg.add(line[2].split("=")[1])
                elif line[-1] == "priorpolarity=positive":
                    spos.add(line[2].split("=")[1])
                elif line[-1] == "priorpolarity=neutral":
                    sneu.add(line[2].split("=")[1])
                elif line[-1] == "priorpolarity=both":
                    spos.add(line[2].split("=")[1])
                    sneg.add(line[2].split("=")[1])

    return wneg, wpos, wneu, sneg, spos, sneu

def bias_lexicon(tokens):
    bias_words = load_bias_lexicon()
    bias_count = float(sum([tokens.count(b) for b in bias_words])) / len(tokens)
    return [bias_count]

def subjective_lexicon(pos_tags):
    subjective_dict = load_subj_lexicon()
    total_count = 0   
    for item in pos_tags:
        if item[0] in subjective_dict.keys():
            if subjective_dict[item[0]][0] == switcher(item[1]) or subjective_dict[item[0]][0] == 'anypos':
                total_count = total_count +1
    return [total_count/len(pos_tags)]

def mpqa_sentiment(tokens):
    wneg, wpos, wneu, sneg, spos, sneu = load_sentiment()
    wneg_count = float(sum([tokens.count(n) for n in wneg])) / len(tokens)
    wpos_count = float(sum([tokens.count(n) for n in wpos])) / len(tokens)
    wneu_count = float(sum([tokens.count(n) for n in wneu])) / len(tokens)
    sneg_count = float(sum([tokens.count(n) for n in sneg])) / len(tokens)
    spos_count = float(sum([tokens.count(n) for n in spos])) / len(tokens)
    sneu_count = float(sum([tokens.count(n) for n in sneu])) / len(tokens)

    return [wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count]


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
