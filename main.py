import os
import xml.sax
import lxml.sax
import lxml.etree
from text_featurizer import *
from myFeaturizer import Featurizer
import pickle
from train_model import parseFeatures

if __name__ == '__main__':
    # read evaluation data
    valFile = "./data/articles-validation-20180831.xml/sampleVal.xml"
    outputFile = './features/val_feat.txt'
    # extract features
    with open(outputFile, 'w') as outFile:
        with open(valFile) as inputFile:
            xml.sax.parse(inputFile, Featurizer(outFile))

    # predict
    ids, valX = parseFeatures('val_feat.txt')
    model_name = './models/lr.sav'
    model = pickle.load(open(model_name, 'rb'))
    preds = model.predict(valX)

    # write output
    predictionFile = 'predictions/predictions.txt'
    with open(predictionFile, 'w') as outFile:
        for i in range(len(ids)):
            outFile.write(ids[i] + " " + preds[i] + " " + "\n")

