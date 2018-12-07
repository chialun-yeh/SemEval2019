#!/usr/bin/env python

"""Extract text from the xml for easier processing"""
import re
import xml.sax
import lxml.sax
import lxml.etree



########## ARTICLE HANDLING ##########
def handleArticle(article, outFile):
    # get text from article
    text = lxml.etree.tostring(article, method="text", encoding="unicode")
    text = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", text)
    text = re.sub(r'[„“]|(\'\')|(,,)', '"', text)
    text = re.sub(r'(www\S+)|(http\S+)|(href)|(\S+.com)', '', text)
    text = re.sub(r'\{.+\} | \[.+\]', '', text)
    text = text.strip('Getty Images').strip('Getty')
    text = re.sub(r'@\S+ | #\S+', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.lstrip().replace('\n','')
    outFile.write(article.get("id") + ',')
    outFile.write(text)
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
    with open(outputFile, 'w', encoding='utf-8') as outFile:
        with open(inputFile, 'r', encoding='utf-8') as inputRunFile:
            parser = xml.sax.make_parser()
            parser.setContentHandler(HyperpartisanNewsExtractor(outFile))
            source = xml.sax.xmlreader.InputSource()
            source.setByteStream(inputRunFile)
            parser.parse(source)