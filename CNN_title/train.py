import os
import numpy as np
import argparse
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
from tensorboardX import SummaryWriter
import pandas as pd

from model import CNN
import utils

def train(data, params):
    
    log_path = 'tensorboard/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    if params["MODEL"] != "rand":
        wv_matrix = []

        if 'glove' in params['WV_FILE']:
            df = pd.read_csv(params['WV_FILE'], sep=" ", quoting=3, header=None, index_col=0)
            glove = {key: val.values for key, val in df.T.items()}
            dim = df.shape[1]
            for i in range(len(data["vocab"])):
                word = data["idx_to_word"][i]
                if word in glove.keys():
                    wv_matrix.append(glove[word])
                else:
                    wv_matrix.append(np.random.uniform(-0.01, 0.01, dim).astype("float32"))

        else:
            word_vectors = KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin", binary=True)   
            dim = len(word_vectors.word_vec('the') )
            for i in range(len(data["vocab"])):
                word = data["idx_to_word"][i]
                if word in word_vectors.vocab:
                    wv_matrix.append(word_vectors.word_vec(word))
                else:
                    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))


        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, dim).astype("float32"))
        wv_matrix.append(np.zeros(dim).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    model = CNN(**params)
    if torch.cuda.is_available():
        model.cuda()
    print('use_cuda = {}\n'.format(torch.cuda.is_available()))
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    best_loss = 1e5
    best_epoch = 0
    num_iter_per_epoch = len(range(0, len(data["train_x"]), params["BATCH_SIZE"]))
        
    for epoch in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for idx, i in enumerate(range(0, len(data["train_x"]), params["BATCH_SIZE"])):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)
            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            if torch.cuda.is_available():
                batch_x = Variable(torch.LongTensor(batch_x)).cuda()
                batch_y = Variable(torch.LongTensor(batch_y)).cuda()
            else:
                batch_x = Variable(torch.LongTensor(batch_x))
                batch_y = Variable(torch.LongTensor(batch_y))


            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

            train_acc = utils.get_evaluation(batch_y.cpu().numpy(), pred.cpu().detach().numpy(), list_metrics=["accuracy"])
            writer.add_scalar('Train/Loss', loss, epoch*num_iter_per_epoch + idx)
            writer.add_scalar('Train/Accuracy', train_acc['accuracy'], epoch*num_iter_per_epoch + idx)

            if (i+1) % 50 == 0:             
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}".format(
                epoch + 1, params['EPOCH'], idx + 1, num_iter_per_epoch, optimizer.param_groups[0]['lr'], loss))
            

        dev_pred = evaluate(data, model, params, 'dev')
        test_pred = evaluate(data, model, params, 'test')
        losses = []

        loss = criterion(dev_pred, data['dev_y'])
        losses.append(loss)
        dev_metrics = utils.get_evaluation(data['dev_y'], dev_pred, list_metrics=["accuracy"])
        writer.add_scalar('Dev/Loss', loss, epoch)
        writer.add_scalar('Dev/Accuracy', dev_metrics["accuracy"], epoch)
        print("Epoch: {}/{}, Loss: {}, Accuracy: {}".format(
                epoch + 1, params['EPOCH'], loss, dev_metrics["accuracy"]))

        test_metrics = utils.get_evaluation(data['test_y'], test_pred, list_metrics=["accuracy"])
        writer.add_scalar('Test Accuracy', test_metrics["accuracy"], epoch)

        if best_loss > loss:
            if params["SAVE_MODEL"]:
                utils.save_model(model, params)
            best_epoch = epoch
            best_loss = loss

        if epoch - best_epoch > 10 and params['EARLY_STOPPING']:
            print('early stopping')
            break


def evaluate(data, model, params, mode):
    model.eval()
    if mode == 'dev':
        x = data["dev_x"]
    else:
        x = data['test_x']

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    if torch.cuda.is_available():
        x = Variable(torch.LongTensor(x)).cuda()
    else:
        x = Variable(torch.LongTensor(x))

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    return pred

def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--model", default="static", help="available models: rand, static, non-static")
    parser.add_argument("--save_model", default=True, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--word_vector_file", default='../data/glove.6B.200d.txt', type=str, help="word vector file path")
    parser.add_argument('--datapath', default = '../', type=str)
    # attempt to speed up
    parser.add_argument("--sample_num", default=1000, type=int, help="training samples that is used")
    parser.add_argument("--max_sent_len", default=20, type=int)
    options = parser.parse_args()

    data = utils.readFile(options.datapath, options.max_sent_len, options.sample_num)
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "WV_FILE": options.word_vector_file,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"]]),
        "BATCH_SIZE": options.batch_size,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "WORD_DIM": 200,
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])

    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    train(data, params)

if __name__ == "__main__":
    main()
