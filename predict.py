import os
import pickle
import xml.sax
from myFeaturizer import Featurizer
from train import parseFeatures
from scipy.sparse import hstack
from gensim import corpora
from doc_representation import extract_doc_rep

if __name__ == "__main__":
    
    use_features = False
    test = False

    run_name = 'val'
    valFile = 'data/articles-validation-bypublisher.xml'
    valDoc = 'features/val_doc.txt'

    # doc representation
    if test:
        dct = corpora.Dictionary.load('tmp/sample/dictionary.dict')  
    else:
        dct = corpora.Dictionary.load('tmp/dictionary.dict')  
    X_rep = extract_doc_rep(valDoc, 'tfidf', run_name, test)

    if use_features:
        feature_path = './features/'
        # extract features
        with open(os.path.join(feature_path, run_name) + '.txt', 'w') as outFile:
            with open(valFile, encoding='utf-8') as inputFile:
                parser = xml.sax.make_parser()
                parser.setContentHandler(Featurizer(outFile))
                source = xml.sax.xmlreader.InputSource()
                source.setByteStream(inputFile)
                source.setEncoding('utf-8')
                parser.parse(source)

        ids, feats = parseFeatures(run_name + '.txt')
        X_val = hstack(X_rep.transpose(), feats)

    else:
        ids = [line.split(',')[0] for line in open(valDoc)]
        X_val = X_rep.transpose()
        
    # predict 
    model_path = './models/'
    model_name = 'LogisticRegression.sav'
    model = pickle.load(open(os.path.join(model_path, model_name), 'rb'))
    preds = model.predict(X_val)

    # write output
    predictionFile = 'predictions/predictions.txt'
    with open(predictionFile, 'w') as outFile:
        for i in range(len(ids)):
            outFile.write(ids[i] + " " + preds[i] + " " + "\n")

