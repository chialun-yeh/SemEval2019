# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python

"""Extract text from the xml for easier processing. This would not be necessary if a way to iterate XML is implemented"""
import re
import xml.sax
import lxml.sax
import lxml.etree
from urllib.request import urlopen


def handleArticle(article, outFile):
    ''' Clean the text by replacing weird quotation marks, removing URLs, bracketed text, news-specific phrases, hashtags and spaces.
        The capitalization and punctuation remain.
    '''
    # get text from article
    text = lxml.etree.tostring(article, method="text", encoding="unicode")
    # clean quotations
    text = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", text)
    text = re.sub(r'[„“]|(\'\')|(,,)', '"', text)
    # remove URLs
    text = re.sub(r'(www\S+)|(http\S+)|(href)|(\S+.com)', '', text)
    # remove anything within {} or [] or ().
    text = re.sub(r'\{[^}]*\}|\[[^]]*\]|\([^)]*\)', '', text)
    # remove irrelevant news usage
    text = re.sub(r'Getty (I|i)mages?|Getty', '', text)
    # remove @ or # tags or amp or weird ......
    text = re.sub(r'@\S+|#\S+|\&?amp|\.{2,}', '', text)
    # remove multiple white spaces
    text = re.sub(r' {2,}', ' ', text)
    # remove newline in the beginning of the file
    text = text.lstrip().replace('\n','')
    outFile.write(article.get("id") + ',')
    outFile.write(text)
    outFile.write("\n")

########## SAX FOR STREAM PARSING ##########
class TextEctractor(xml.sax.ContentHandler):
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
    with open(outputFile, 'w', encoding='utf-8') as outFile:
        with urlopen(inputFile) as inputRunFile:
            parser = xml.sax.make_parser()
            parser.setContentHandler(TextEctractor(outFile))
            source = xml.sax.xmlreader.InputSource()
            source.setByteStream(inputRunFile)
            source.setEncoding('utf8')
            parser.parse(source)



if __name__ == '__main__':
    inputFile = "https://s3.amazonaws.com/sbd2018/SemEval_data/articles-validation-bypublisher.xml"
    outputFile = 'text/test.txt'
    createDocuments(inputFile, outputFile)



