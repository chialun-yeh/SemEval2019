#!/usr/bin/env python
import xml.sax

class GroundTruthHandler(xml.sax.ContentHandler):
    def __init__(self, gt, bias, urls):
        xml.sax.ContentHandler.__init__(self)
        self.groundTruth = gt
        self.bias = bias
        self.urls = urls

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id")
            hyperpartisan = attrs.getValue("hyperpartisan")
            if hyperpartisan in self.groundTruth.keys():
                self.groundTruth[hyperpartisan] = self.groundTruth[hyperpartisan] + 1
            else:
                self.groundTruth[hyperpartisan] = 1

            if 'bias' in attrs:
                bia = attrs.getValue("bias")
                if bia in self.bias.keys():
                    self.bias[bia] = self.bias[bia] + 1
                else:
                    self.bias[bia] = 1
            
            url = attrs.getValue("url")
            # parse url
            url = '/'.join(url.split('/')[:3])
            if url in self.urls.keys():
                self.urls[url].append(hyperpartisan)
            else:
                self.urls[url] = [hyperpartisan]
            
            

def testClass(gt_file):
    gt, bias, urls = {}, {}, {}
    with open(gt_file) as f:
        xml.sax.parse(f, GroundTruthHandler(gt, bias, urls))

    same, diff = 0,0
    single = 0
    for key, value in urls.items():
        if len(value) > 1:
            if( len(set(value)) <= 1 ):
                same += 1
            else: 
                diff += 1
        else:
            single += 1
    urls_with_multiple = same + diff

    print('Hyperpartisan: ', gt)
    if bias:
        print('Bias: ', bias)
    print('URLs with same label: %i | different label: %i | Total URLs with multiple articles: %i | URLs with a single article: %i' \
    %(same, diff, urls_with_multiple, single))

                           
if __name__ == '__main__':
    # Parse groundTruth
    yTst = "../data/ground-truth-training-byarticle.xml"
    yTrn = "../data/ground-truth-training-bypublisher.xml"
    yVal = "../data/ground-truth-validation-bypublisher.xml"

    data ={'Train':yTrn, 'Develop':yVal, 'Test':yTst}       
    for key, value in data.items():
        print('='*20 + key + '='*20)
        testClass(value)






    
    

