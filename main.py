import os
import xml.sax
import lxml.sax
import lxml.etree
from text_featurizer import *
from myFeaturizer import Featurizer
import pickle
from train_model import parseFeatures
from bow import *

def run_prediction(test, use_features, dimension):
    if test:
        name = 'sample_val'
        valFile = 'data/sampleArticle_val.xml'
        labelFile = 'data/ground_val.xml'
    else:
        name = 'val'
        valFile = 'data/articles-validation-bypublisher.xml'
        labelFile = 'data/ground-truth-validation-bypublisher.xml'

    # doc representation
    dic = corpora.Dictionary.load('tmp/dictionary.dict')  
    X_rep = extract_doc_rep(valFile, dic, dimension, 'bow')

    if use_features:
        feature_path = './features/'
        # extract features
        with open(os.path.join(feature_path, name) + '.txt', 'w') as outFile:
            with open(valFile, encoding='utf-8') as inputFile:
                parser = xml.sax.make_parser()
                parser.setContentHandler(Featurizer(outFile))
                source = xml.sax.xmlreader.InputSource()
                source.setByteStream(inputFile)
                source.setEncoding('utf-8')
                parser.parse(source)

        ids, feats = parseFeatures(name + '.txt')
        X_val = np.hstack(np.transposr(X_rep), feats)

    else:
        ids = [line.split(',')[0] for line in open(allDocs)]
        X_val = np.transpose(X_rep)
        

    # predict 
    model_name = './models/lr.sav'
    model = pickle.load(open(model_name, 'rb'))
    preds = model.predict(X_val)

    # write output
    predictionFile = 'predictions/predictions.txt'
    with open(predictionFile, 'w') as outFile:
        for i in range(len(ids)):
            outFile.write(ids[i] + " " + preds[i] + " " + "\n")

