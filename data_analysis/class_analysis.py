#!/usr/bin/env python

from __future__ import division
import xml.sax

binary = {}
bias = {}
label = {}
class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if name == "article":
            b = attrs.getValue("hyperpartisan")
            if b in binary.keys():
                binary[b] = binary[b] + 1
            else:
                binary[b] = 0

            if 'bias' in attrs:
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
    yTst = "data/ground-truth-training-byarticle.xml"
    yTrn = "data/ground-truth-training-bypublisher.xml"
    yVal = "data/ground-truth-validation-bypublisher.xml"
    with open(yTrn) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())
        print(binary)
        print(bias)
        binary = {}
        bias = {}
    with open(yVal) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())
        print(binary)
        print(bias)
        binary = {}
        bias = {}
    with open(yTst) as groundTruthDataFile:
        xml.sax.parse(groundTruthDataFile, GroundTruthHandler())
        print(binary)

    

