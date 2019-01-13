"""
@author: chialunyeh
"""
import os
import getopt
import sys
import numpy as np
import torch
import torch.nn.functional as F
import utils

model = utils.load_model("saved_model/")
word2vec_path = "../data/glove.6B.200d.txt"
outputFile = "predictions.txt"
 
def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)


def test(inputFile, outFile):

    data = utils.readFile()
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    word_vectors = KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin", binary=True)
    wv_matrix = []
    for i in range(len(data["vocab"])):
        word = data["idx_to_word"][i]
        if word in word_vectors.vocab:
            wv_matrix.append(word_vectors.word_vec(word))
        else:
            wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    wv_matrix.append(np.zeros(300).astype("float32"))
    wv_matrix = np.array(wv_matrix)
    params["WV_MATRIX"] = wv_matrix

    

    
    x = data["test_x"]
    x = [[data["word_to_idx"][w] if w in data["vocab"] else len(data["vocab"]) for w in sent] +
         [len(data["vocab"]) + 1] * 20 - len(sent))]

    if torch.cuda.is_available():
        x = Variable(torch.LongTensor(x)).cuda()
    else:
        x = Variable(torch.LongTensor(x))

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)

    for i, j in zip(te_index, pred):
        if np.argmax(j) == 0:
            outFile.write(str(i) + " true" + "\n")
        else:
            outFile.write(str(i) + " false" + "\n")

            
def main(inputDataset, outputDir):
    """Main method of this module."""
    tmp_file_name = 'tmp.txt'
    with open(outputDir + "/" + outputFile, 'w', encoding = 'utf-8') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                createDocuments(inputDataset + '/' + file, tmp_file_name)
                test(tmp_file_name, outFile)
    
    if os.path.exists(tmp_file_name):
        os.remove(tmp_file_name) 


class Predictor(xml.sax.ContentHandler):
    def __init__(self, outFile):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile
        self.lxmlhandler = "undefined"

    def startElement(self, name, attrs):
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()
            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                # pass to handleArticle function
                handleArticle(self.lxmlhandler.etree.getroot(), self.outFile)                
                self.lxmlhandler = "undefined"


########## MAIN #########
def main(inputDataset, outputDir):
    """Main method of this module."""

    with open(outputDir + "/" + runOutputFileName, 'w', encoding = 'utf8') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                with open(inputDataset + "/" + file, 'r', encoding = 'utf8') as inputRunFile:
                    parser = xml.sax.make_parser()
                    parser.setContentHandler(Predictor(outFile))
                    source = xml.sax.xmlreader.InputSource()
                    source.setByteStream(inputRunFile)
                    parser.parse(source)


if __name__ == '__main__':
    #main(*parse_options())
    main('../data/article', 'predictions')
