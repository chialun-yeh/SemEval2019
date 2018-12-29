import os
import xml.sax
import lxml.sax
import lxml.etree
import numpy as np
from xml.etree import cElementTree as ET

dataFile = 'data/articles-training-byarticle.xml'


class Extractor(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
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
                art = self.lxmlhandler.etree.getroot()  
                
                self.lxmlhandler = "undefined"
                
from xml.etree import cElementTree as ET
for event, elem in ET.iterparse(dataFile):
    if elem.tag != 'articles':
        print(elem.text)
        elem.clear()