import torch
import xml.etree.ElementTree as ET
from sklearn import metrics
import numpy as np

def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def readFile(datapath, max_sent_length=20, sample_num=1000000):
    data = {}
    max_sent_length = max_sent_length
    sample_num = sample_num
    dataDir = datapath

    def read(mode, name):
        idx_check = []
        x, y = [], []

        with open(dataDir + "data/ground-truth-" + name + ".xml", encoding="utf-8") as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for article in root.iter('article'):
                idx_check.append(article.attrib['id'])
                y.append(article.attrib['hyperpartisan'])

        idx_check_iter = iter(idx_check)
        with open(dataDir + "data/articles-" + name + ".txt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                articleId = line.split('::')[0]
                title = line.split('::')[1].split()
                x.append(title[:max_sent_length])

        if mode == "train":
            data["train_x"], data["train_y"] = x[:sample_num], y[:sample_num]
        elif mode == 'dev':
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("train", 'training-bypublisher')
    read('dev', 'validation-bypublisher')
    #read("train", 'training-byarticle')
    #read('dev', 'training-byarticle')
    read('test', 'training-byarticle')

    return data

def save_model(model, params):
    path = f"saved_models/{params['MODEL']}_{params['EPOCH']}_{params['BATCH_SIZE']}.pkl"
    torch.save(model, path)
    print(f"A model is saved successfully as {path}!")


def load_model(path):
    try:
        model = torch.load(path)
        print("Model loaded successfully!")
        return model
    except:
        print(f"No available model such as {path}.")
        exit()

if __name__ == "__main__":
    data = readFile("../",10,100)
    print(data['train_x'][0], len(data['train_x']))
    print(data['dev_x'][0], len(data['dev_x']))