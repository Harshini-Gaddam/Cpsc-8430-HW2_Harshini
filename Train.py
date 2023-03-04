#!/usr/bin/env python
# coding: utf-8

# In[3]:


import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.special import expit
import random
import sys
import json
import re
from torch.utils.data import DataLoader, Dataset
import pickle
import models
from torch.optim.lr_scheduler import ReduceLROnPlateau


Data_path='/Users/harshini/Documents/DL_HW_2/MLDS_hw2_1_data'
Created_Model_path='Created_Model'
pickel_file='Created_Model/pickel_data.pickle'


# In[14]:



class Dataprocessor(Dataset):
    def __init__(self, label_file, files_dir,dictonary, w2i):
    #def __init__(self):
        self.label_file = label_file
        self.files_dir = files_dir
        self.avi = avi(label_file)
        self.w2i = w2i
        self.dictonary = dictonary
        self.data_pair = annotate(files_dir, dictonary, w2i)
    def __len__(self):
        return len(self.data_pair)
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000
        return torch.Tensor(data), torch.Tensor(sentence)

class test_dataloader(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]
    
def dictonaryFunc(word_min): 
#Json Loader#
    with open('training_label.json', 'r') as f:
        file = json.load(f)

    word_count = {}
    for d in file:
        for s in d['caption']:
            words_in_sentence = re.sub('[.!,;?]]', ' ', s).split()
            for word in words_in_sentence:
                word = word.replace('.', '') if '.' in word else word
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

                    
#Creating Dictonary using tokens    
    dictonary_1 = {}
    for word in word_count:
        if word_count[word] > word_min:
            dictonary_1[word] = word_count[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(dictonary_1)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(dictonary_1)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
    return i2w,w2i,dictonary_1


def spliting_sentence(sentence, dictonary_1, w2i):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in dictonary:
            sentence[i] = 3
        else:
            sentence[i] = w2i[sentence[i]]
    sentence.insert(0, 1)
    sentence.append(2)
    return sentence


def annotate(label_file, dictonary_1, w2i):
    label_json = label_file
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = spliting_sentence(s, dictonary, w2i)
            annotated_caption.append((d['id'], s))
    return annotated_caption

def word_to_index(w2i, w):
        return w2i[w]
def index_to_word(i2w, i):
    return i2w[i]
def sentence_to_index(w2i, sentence):
    return [w2i[w] for w in sentence]
def index_to_sentence(i2w, index_seq):
    return [i2w[int(i)] for i in index_seq]


def avi(files_dir):
    avi_data = {}
    training_feats = files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data


# Train Function
def train(model, epoch, train_loader, loss_func):
    model.train()
    print(epoch)
    model = model
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001)
    running_loss=0.0
    
    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = avi_feats, ground_truths
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)
        
        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
            
        ground_truths = ground_truths[:, 1:]  
        loss = calculate_loss(seq_logProb, ground_truths, lengths,loss_func)
        loss.backward()
        optimizer.step()
        
    loss = loss.item()
    
    running_loss += loss.item()
        if batch_idx % 10 == 9:
            print(f"Batch: {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/10:.3f}")
            running_loss=0.0
    
    print(f'Epoch:{epoch} & loss:{np.round(loss,3)}')

        
#Evaluate
def evaluate(test_loader,model):
    # set model to evaluation(testing) mode
    model.eval()
    test_predictions, test_truth = None, None
    for batch_idx, batch in enumerate(test_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats, ground_truths = avi_feats, ground_truths
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

        seq_logProb, seq_predictions = model(avi_feats, mode='inference')
        ground_truths = ground_truths[:, 1:]
        test_predictions = seq_predictions[:3]
        test_truth = ground_truths[:3]
        break


#TestFunction
def test(test_loader,model,i2w):
        
        model.eval()
        ss = []
        for batch_idx, batch in enumerate(test_loader):
            # prepare data
            id, avi_feats = batch
            avi_feats=avi_feats
                
            id, avi_feats = id, Variable(avi_feats).float()
            
            seq_logProb, seq_predictions = model(avi_feats, mode='inference')
            test_predictions = seq_predictions

            result = [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
            result = [' '.join(s).split('<EOS>')[0] for s in result]
        
            rr = zip(id, result)
            for r in rr:
                ss.append(r)
        return ss

#Loss Function
def calculate_loss(x, y, lengths,loss_fn):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] -1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)
    avg_loss = loss/batch_size

    return loss


#Minibatch

def minibatch(data):

    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths


# In[15]:


def main():

    label_file = 'training_data/feat'
    files_dir = 'training_label.json'
    i2w,w2i,dictonary = dictonaryFunc(4)
    train_dataset = Dataprocessor(label_file, files_dir,dictonary, w2i)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=10, shuffle=True, num_workers=4, collate_fn=minibatch)
    
    label_file = 'testing_data/feat'
    files_dir = 'testing_label.json'
    test_dataset = Dataprocessor(label_file, files_dir,dictonary, w2i)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=minibatch)
   
    epochs_n = 25
    ModelSaveLoc = 'Created_Model'
    with open('pickel_data.pickle', 'wb') as f:
         pickle.dump(i2w, f)

    
    x = len(i2w)+4
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)
    loss_fn = nn.CrossEntropyLoss()
    encode = models.EncoderNet()
    decode = models.DecoderNet(512, x, x, 1024, 0.3)
    model = models.ModelMain(encoder = encode,decoder = decode) 
    

    start = time.time()
    for epoch in range(epochs_n):
        train(model,epoch+1, train_loader=train_dataloader, loss_func=loss_fn)
        evaluate(test_dataloader, model)

    end = time.time()
    torch.save(model, "{}/{}.h5".format(ModelSaveLoc, 'model0'))
    print("Training finished {}  elapsed time: {: .3f} seconds. \n".format('test', end-start))
    


# In[16]:


if __name__=="__main__":
    main()


# In[ ]:




