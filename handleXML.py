#!/usr/bin/env python

"""Extract text from the xml for easier processing"""

import os
import getopt
import sys
import xml.sax
import lxml.sax
import lxml.etree
import re


########## ARTICLE HANDLING ##########
def handleArticle(article, outFile):
    # get text from article
    text = lxml.etree.tostring(article, method="text", encoding="unicode")
    remove = re.sub('\([a-z]*\)|(\#\S+)', '', text.lower())
    remove_url = re.sub('(www\S+)|(http\S+)|(href)|(\S+.com)', '', remove)
    textcleaned = re.sub('[^a-z ]', '', remove_url)
    textcleaned = re.sub('(^ *) | ( *$)', '', textcleaned)
    textcleaned = re.sub('(  +)', ' ', textcleaned)
    outFile.write(article.get("id") + ',')
    outFile.write(textcleaned)
    outFile.write("\n")

########## SAX FOR STREAM PARSING ##########
class HyperpartisanNewsExtractor(xml.sax.ContentHandler):
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

def createDocuments(inputFile, outputFile):
    with open(outputFile, 'w') as outFile:
        with open(inputFile, 'r', encoding='utf-8') as inputRunFile:
            parser = xml.sax.make_parser()
            parser.setContentHandler(HyperpartisanNewsExtractor(outFile))
            source = xml.sax.xmlreader.InputSource()
            source.setByteStream(inputRunFile)
            source.setEncoding("utf-8")
            parser.parse(source)

