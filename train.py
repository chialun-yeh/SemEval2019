import sys
import os
import getopt
import pickle
import xml.sax
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from myFeaturizer import Featurizer, parseFeatures
from doc_representation import buildRep, extract_doc_rep
from handleXML import createDocuments
from scipy.sparse import hstack


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

    return (inputFile, labelFile, outputPath, rep, model)


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
    if model == 'lr':
        model = LogisticRegression()
        model_name = 'logisticRegression.sav'

    elif model == 'rf':
        model = RandomForestClassifier()
        model_name = 'randomForest.sav'
    print('training model')
    model.fit(X, Y)
    pickle.dump(model, open(os.path.join(outputPath, model_name),'wb'))

def main(inputFile, labelFile, outputPath, rep, model):  
    """Main method of this module."""

    use_features = False
    # parse groundTruth
    with open(labelFile) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())
    
    # build document representation model
    if not os.path.exists(inputFile.strip('.xml') + '.txt'):
        createDocuments(inputFile)
    else:
        print('loading documents')
        # also create for validation?
    docs = inputFile.strip('.xml') + '.txt'
    if rep == 'bow' or rep == 'tfidf':
        dim = 50000
    else:
        dim = 300

    buildRep(docs, rep, dim)
    # extract doc representation for training set
    X_rep = extract_doc_rep(docs, rep, name='trn_rep')

    if use_features:
        feature_path = './features/'
        # extract and write features to ./features/
        with open(os.path.join(feature_path, inputFile.strip('.xml')) + '.txt', 'w') as outFile:
            with open(inputFile, encoding='utf-8') as inputFile:
                parser = xml.sax.make_parser()
                parser.setContentHandler(Featurizer(outFile))
                source = xml.sax.xmlreader.InputSource()
                source.setByteStream(inputFile)
                source.setEncoding('utf-8')
                parser.parse(source)
        # parse features
        ids, feats = parseFeatures(os.path.join(feature_path, inputFile.strip('.xml')) + '.txt')
        X_trn = hstack(X_rep.transpose(), feats)

    else:
        X_trn = X_rep.transpose()
        ids = [line.split(',')[0] for line in open(docs)]
    
    # build label for training
    Y_trn = []
    for i in ids:
        Y_trn.append(groundTruth[i])

    # train model
    train_model(X_trn, Y_trn, model, outputPath)


if __name__ == '__main__':
    main(*parse_options())

    

    

