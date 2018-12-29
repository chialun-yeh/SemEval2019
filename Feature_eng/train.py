import sys
import os
import getopt
import argparse
import pickle
import xml.sax
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from scipy import sparse
import numpy as np

from myFeaturizer import Featurizer, parseFeatures
from doc_representation import buildRep, extract_doc_rep
from handleXML import createDocuments



########## OPTIONS HANDLING ##########
def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputFile=", "labelFile=", "outputPath=", "documentRep=", "model="]
        opts, _ = getopt.getopt(sys.argv[1:], "i:l:o:d:m:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputFile = "undefined"
    labelFile = "undefined"
    outputPath = "undefined"
    rep = 'bow'
    model = 'lr'
    
    for opt, arg in opts:
        if opt in ("-i", "--inputFile"):
            inputFile = arg
        elif opt in ("-l", "--labelFile"):
            labelFile = arg
        elif opt in ("-o", "--outputPath"):
            outputPath = arg
        elif opt in ("-d", "--documentRep"):
            rep = arg
        elif opt in ("-m", "--model"):
            model = arg
        else:
            assert False, "Unknown option."
    if inputFile == "undefined":
        sys.exit("The input XML file of aritcles, is undefined. Use option -i or --inputFile.")
    elif not os.path.exists(inputFile):
        sys.exit("The input file does not exist (%s)." % inputFile)

    if labelFile == "undefined":
        sys.exit("The label XML file is undefined. Use option -l or --labelFile.")
    elif not os.path.exists(inputFile):
        sys.exit("The label file does not exist (%s)." % labelFile)

    if outputPath == "undefined":
        sys.exit("The output path where the model should be saved, is undefined. Use option -o or --outputFile.")
    elif not os.path.exists(outputPath):
        os.mkdir(outputPath)

    if rep not in ['bow', 'tfidf', 'doc2vec']:
        sys.exit("The supported document representations are BOW, TFIDF, or Doc2Vec")

    if model not in ['lr', 'rf']:
        sys.exit('The supported models are logistic regression (lr) or random forest (rf)')

    return (inputFile, labelFile, outputPath)


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

def train_model(X, Y, model, outputPath):
    """Train main model with input features X and label Y."""
    if model == 'lr':
        model = LogisticRegression()
        model_name = 'logisticRegression.sav'

    elif model == 'rf':
        model = RandomForestClassifier()
        model_name = 'randomForest.sav'
    print('training model')
    model.fit(X, Y)
    pickle.dump(model, open(outputPath + '/' + model_name,'wb'))

def main(inputFile, labelFile, outputPath):  
    
    """Main method of this module."""

    ''' 
    Pre-processing for word-based method: lower everything, strip all punctuation, lemmatize/stemming
    Word-based representation: BOW, TFIDF, ngrams 
    Learned represenatation: doc2vec

    Tunable: 
    [I] document representation
        1. stem or not
        2. bow/tfidf: vector_size
        3. doc2vec: vector_size, window, min_count
    [II] additional features
        1. title: 
        2. text
    [III] models
        1. algorithm: LR, RF, NB, SVM
        2. parameters of the algorithm
    '''

    stem = False
    rep = 'doc2vec'
    dim = 100
    use_title = False
    use_features = False
    model = 'lr'

    
    # parse groundTruth
    with open(labelFile) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())
    
    # build document representation model
    text_folder = 'text/'
    if not os.path.exists(text_folder):
        os.mkdir(text_folder)
    text_name = Path(inputFile).name.strip('.xml') + '.txt'
    if not os.path.exists(text_folder + text_name ):
        print('creating text document...')
        createDocuments(inputFile, text_folder + text_name)
    else:
        print('loading text document...')
        # also create for validation?
    if rep == 'bow' or rep == 'tfidf':
        dim = 50000
    else:
        dim = 100

    buildRep(text_folder, rep, dim, stem)
    # extract doc representation for training set
    X_rep = extract_doc_rep(text_folder + text_name, rep, dim, stem)

    if use_features or use_title:
        feature_path = 'features/'
        if not os.path.exists(feature_path):
            os.mkdir(feature_path)
        if not os.path.exists(feature_path + text_name ):
            print('computing features....')
            # extract and write features to ./features/
            with open(feature_path + text_name, 'w') as outFile:
                with open(inputFile, 'r', encoding='utf-8') as inputFile:
                    parser = xml.sax.make_parser()
                    parser.setContentHandler(Featurizer(outFile, use_features))
                    source = xml.sax.xmlreader.InputSource()
                    source.setByteStream(inputFile)
                    parser.parse(source)

        # parse features
        articleIds, feats = parseFeatures(text_name)
        if rep == 'doc2vec':
            X_trn = np.hstack( (X_rep.transpose(), feats))
        else:
            X_trn = sparse.hstack( (X_rep.transpose(), feats))

    else:
        X_trn = X_rep.transpose()
        articleIds = [line.split(',')[0] for line in open(text_folder + text_name, 'r', encoding='utf8')]
    
    X_trn = normalize(X_trn)
    # build label for training
    Y_trn = []
    for articleId in articleIds:
        Y_trn.append(groundTruth[articleId])

    # train model
    train_model(X_trn, Y_trn, model, outputPath)


if __name__ == '__main__':
    inputFile = 'data/articles-training-bypublisher.xml'
    labelFile = 'data/ground-truth-training-bypublisher.xml'
    outputPath = 'model'
    main(inputFile, labelFile, outputPath)
    #main(*parse_options())