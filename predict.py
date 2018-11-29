import os
import pickle
import xml.sax
from myFeaturizer import Featurizer
from train import parseFeatures
from scipy.sparse import hstack
from doc_representation import *

if __name__ == "__main__":
    
    test = False
    use_features = False

    if test:
        name = 'sample_val'
        valFile = 'data/sampleArticle_val.xml'
        labelFile = 'data/ground_val.xml'
    else:
        name = 'val'
        valFile = 'data/articles-validation-bypublisher.xml'
        labelFile = 'data/ground-truth-validation-bypublisher.xml'

    valDoc = 'features/val_doc.txt'

    # doc representation
    dic = corpora.Dictionary.load('tmp/dictionary.dict')  
    X_rep = extract_doc_rep(valDoc, dic, 'bow', name='val_rep')

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
        X_val = hstack(X_rep.transpose(), feats)

    else:
        ids = [line.split(',')[0] for line in open(valDoc)]
        X_val = X_rep.transpose()
        
    # predict 
    model_name = './models/LogisticRegression.sav'
    model = pickle.load(open(model_name, 'rb'))
    preds = model.predict(X_val)

    # write output
    predictionFile = 'predictions/predictions.txt'
    with open(predictionFile, 'w') as outFile:
        for i in range(len(ids)):
            outFile.write(ids[i] + " " + preds[i] + " " + "\n")

