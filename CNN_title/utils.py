import pickle
import xml.etree.ElementTree as ET
def read_SemEval():
    data = {}

    def read(mode, name):
        idx_check = []
        x, y = [], []

        with open("../data/ground-truth-" + name + ".xml", encoding="utf-8") as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for article in root.iter('article'):
                idx_check.append(article.attrib['id'])
                y.append(article.attrib['hyperpartisan'])

        idx_check_iter = iter(idx_check)
        with open("../data/articles-" + name + ".xml", encoding="utf-8") as f:
            tree = ET.iterparse(f)
            root = tree.getroot()
            for article in root.iter('article'):
                if article.attrib['id'] == next(idx_check_iter):
                    x.append(article.attrib['title'])
        
        assert(len(x) == len(y))

        if mode == "train":
            data["train_x"], data["train_y"] = x, y
        elif mode == 'dev':
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("train", 'training-bypublisher')
    read('dev', 'validation-bypublisher')
    read("test", 'training-byarticle')

    return data



def save_model(model, params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")


def load_model(params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}.")
        exit()
