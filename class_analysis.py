#!/usr/bin/env python

"""Calculates the measures for the PAN19 hyperpartisan news detection task"""
# Version: 2018-09-24

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the ground truth XML file with the articles for which a prediction should have been made.
# --inputRun=<directory>
#   Directory that contains the prediction for the articles in the ground truth XML file. The format of the XML file should be, one article per line:
#     <article id> <prediction> [<confidence>]
#   where:
#     - article id   corresponds to the "id" attribute of the "article" element in the articles and ground truth files
#     - prediction   is either "true" (hyperpartisan) or "false" (not hyperpartisan)
#     - confidence   is an optional value to describe the confidence of the predictor in the prediction---the higher, the more confident. If missing, a value of 1 is used. However, the absolute values are unimportant: this may just be used in the future to order the predictions, for example to calculate ROC curves.
# --outputDir=<directory>
#   Directory to which the evaluation will be written. Will be created if it does not exist.

from __future__ import division

import json
import os
import sys
import xml.sax

########## SAX ##########

binary = {}
bias = {}
label = {}
class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            b = attrs.getValue("hyperpartisan")
            if b in binary.keys():
                binary[b] = binary[b] + 1
            else:
                binary[b] = 0

            bia = attrs.getValue("bias")
            if bia in bias.keys():
                bias[bia] = bias[bia] + 1
            else:
                bias[bia] = 0

            s = attrs.getValue("labeled-by")
            if s in label.keys():
                label[s] = label[s] + 1
            else:
                label[s] = 0





if __name__ == '__main__':
    yTrn = "C:/Users/sharo/Documents/SemEval2019/data/ground-truth-training-20180831.xml/yTrn.xml"
    yVal = "C:/Users/sharo/Documents/SemEval2019/data/ground-truth-validation-20180831.xml/yVal.xml"
    with open(yVal) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())

    print(binary)
    print(bias)
    print(label)

