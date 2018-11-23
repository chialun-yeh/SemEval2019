#!/usr/bin/env python

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

