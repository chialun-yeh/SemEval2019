#!/usr/bin/env python

"""Extract text from the xml for easier processing. This would not be necessary if a way to iterate XML is implemented"""
import re
import xml.sax
import lxml.sax
import lxml.etree

def cleanQuotations(text):
    # clean quotations
    text = re.sub(r'[`‘’‛⸂⸃⸌⸍⸜⸝]', "'", text)
    text = re.sub(r'[„“]|(\'\')|(,,)', '"', text)
    return text

def cleanText(text):  
    # remove URLs
    text = re.sub(r'(www\S+)|(http\S+)|(href)', '', text)
    # remove anything within {} or [] or ().
    text = re.sub(r'\{[^}]*\}|\[[^]]*\]|\([^)]*\)', '', text)
    # remove irrelevant news usage
    text = re.sub(r'Getty [Ii]mages?|Getty|[Ff]ollow us on [Tt]witter|MORE:', '', text)
    # remove @ or # tags or amp or weird ......
    text = re.sub(r'@\S+|#\S+|\&amp|\.{2,}', '', text)
    # remove multiple white spaces
    text = re.sub(r' {2,}', ' ', text)
    # remove newline in the beginning of the file
    text = text.lstrip().replace('\n','')
    return text

def handleArticle(article, outFile):
    ''' Clean the text by replacing weird quotation marks, removing URLs, bracketed text, news-specific phrases, hashtags and spaces.
        The capitalization and punctuation remain.
    '''
    # get text from article
    title = article.get('title')
    title = cleanQuotations(title)
    text = lxml.etree.tostring(article, method="text", encoding="unicode")
    text = cleanQuotations(text)
    text = cleanText(text)
    outFile.write(article.get("id") + '::')
    outFile.write(title + '::')
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
        with open(inputFile, 'r', encoding='utf-8') as inputRunFile:
            parser = xml.sax.make_parser()
            parser.setContentHandler(TextEctractor(outFile))
            source = xml.sax.xmlreader.InputSource()
            source.setByteStream(inputRunFile)
            parser.parse(source)



if __name__ == '__main__':
    inputFile = '../data/articles-training-bypublisher.xml'
    outputFile = '../data/articles-training-bypublisher.txt'
    createDocuments(inputFile, outputFile)