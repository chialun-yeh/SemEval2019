"""
Adapted from https://github.com/1991viet/Hierarchical-attention-networks-pytorch
@author: Viet Nguyen
@author: chialunyeh
"""
import os
import getopt
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import MyDataset
from src.handleXML import createDocuments

batch_size = 128
pre_trained_model = "trained_model/trained_model_han"
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
    test_params = {"batch_size": batch_size,
                   "shuffle": False,
                   "drop_last": False}

    if torch.cuda.is_available():
        model = torch.load(pre_trained_model)
    else:
        model = torch.load(pre_trained_model, map_location=lambda storage, loc: storage)

    test_set = MyDataset(inputFile, word2vec_path, '', model.max_sent_length, model.max_word_length)
    test_generator = DataLoader(test_set, **test_params)
    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    te_index_ls = []
    te_pred_ls = []
    for index, te_feature, te_label in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
            te_predictions = F.softmax(te_predictions)
        te_index_ls.extend(index)
        te_pred_ls.append(te_predictions.clone().cpu())
    te_pred = torch.cat(te_pred_ls, 0).numpy()
    te_index = np.array(te_index_ls)
  
    for i, j in zip(te_index, te_pred):
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


if __name__ == '__main__':
    #main(*parse_options())
    main('../data/article', 'predictions')
