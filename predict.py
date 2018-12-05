from __future__ import division

import os
import sys
import getopt
import pickle
import xml.sax
from myFeaturizer import Featurizer
from train import parseFeatures
from scipy.sparse import hstack
from gensim import corpora
from doc_representation import extract_doc_rep
from handleXML import createDocuments


def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputFile=", "outputDir=", "modelName=", "documentRep="]
        opts, _ = getopt.getopt(sys.argv[1:], "i:o:m:d:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputFile = "undefined"
    outputDir = "undefined"
    modelName = "undefined"
    documentRep = 'bow'

    for opt, arg in opts:
        if opt in ("-i", "--inputFile"):
            inputFile = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        elif opt in ("-m", "--modelName"):
            modelName = arg
        elif opt in ('-d', '--documentRep'):
            documentRep = arg
        else:
            assert False, "Unknown option."
    if inputFile == "undefined":
        sys.exit("The input dataset that is to be predicted is undefined. Use option -i or --inputFile.")
    elif not os.path.exists(inputFile):
        sys.exit("The input dataset does not exist (%s)." % inputFile)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    if modelName == 'undefined':
        sys.exit('The model to perform the prediction is undefined. Use option -m or --modelName')
    elif not os.path.exists(modelName):
        sys.exit("The model does not exist (%s)." % modelName)

    return (inputFile, outputDir, modelName, documentRep)

def main(inputFile, outputDir, modelName, rep):

    use_features = False
    if not os.path.exists(inputFile.strip('.xml') + '.txt'):
        createDocuments(inputFile)
    doc = inputFile.strip('.xml') + '.txt'

    # doc representation
    X_rep = extract_doc_rep(doc, rep)

    if use_features:
        feature_path = './features/'
        # extract features
        with open(os.path.join(feature_path, inputFile.strip('.xml')) + '.txt', 'w') as outFile:
            with open(inputFile, encoding='utf-8') as inputFile:
                parser = xml.sax.make_parser()
                parser.setContentHandler(Featurizer(outFile))
                source = xml.sax.xmlreader.InputSource()
                source.setByteStream(inputFile)
                source.setEncoding('utf-8')
                parser.parse(source)

        ids, feats = parseFeatures(inputFile.strip('.xml') + '.txt')
        X_val = hstack(X_rep.transpose(), feats)

    else:
        ids = [line.split(',')[0] for line in open(doc)]
        X_val = X_rep.transpose()

    # predict 
    print('predicting values...')
    model = pickle.load(open(modelName, 'rb'))
    preds = model.predict(X_val)

    # write output
    predictionFile = outputDir + '/predictions.txt'
    with open(predictionFile, 'w') as outFile:
        for i in range(len(ids)):
            outFile.write(ids[i] + " " + preds[i] + " " + "\n")
    
if __name__ == '__main__':
    main(*parse_options())

    
        
    

