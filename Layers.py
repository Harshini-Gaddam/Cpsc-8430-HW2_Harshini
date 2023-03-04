#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.special import expit
import pickle


# In[ ]:


with open('pickel_data.pickle', 'rb') as f:
    pickeldata_load = pickle.load(f)


# In[ ]:


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, sequence_length, feature_number = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feature_number).repeat(1, sequence_length, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, sequence_length)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context


# In[ ]:


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.4)
        self.gru = nn.GRU(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, sequence_length, feature_number = input.size()    
        input = input.view(-1, feature_number)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, sequence_length, 512)

        output, hidden_state = self.gru(input)

        return output, hidden_state


# In[ ]:


class DecoderNet(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.4):
        super(DecoderNet, self).__init__()

        self.hidden_size = 512
        self.output_size = len(pickeldata_load) + 4 
        self.vocab_size = len(pickeldata_load) + 4
        self.word_dim = 1024

        # layers
        self.embedding = nn.Embedding(len(pickeldata_load) + 4, 1024)
        self.dropout = nn.Dropout(0.4)
        self.gru = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)


    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word
        sequence_logProb = []
        sequence_predictions = []

        targets = self.embedding(targets)
        _, sequence_length, _ = targets.size()

        for i in range(sequence_length-1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold: # returns a random float value between 0.05 and 0.995
                current_input_word = targets[:, i]  
            else: 
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            sequence_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        sequence_logProb = torch.cat(sequence_logProb, dim=1)
        sequence_predictions = sequence_logProb.max(2)[1]
        return sequence_logProb, sequence_predictions
        
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word
        sequence_logProb = []
        sequence_predictions = []
        assumption_sequence_length = 28
        
        for i in range(assumption_sequence_length-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            gru_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            gru_output, decoder_current_hidden_state = self.gru(gru_input, decoder_current_hidden_state)
            logprob = self.to_final_output(gru_output.squeeze(1))
            sequence_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        sequence_logProb = torch.cat(sequence_logProb, dim=1)
        sequence_predictions = sequence_logProb.max(2)[1]
        return sequence_logProb, sequence_predictions

    def teacher_forcing_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85))


# In[ ]:


class ModelMain(nn.Module):
    def __init__(self, encoder, decoder):
        super(ModelMain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            sequence_logProb, sequence_predictions = self.decoder(encoder_last_hidden_state = encoder_last_hidden_state, encoder_output = encoder_outputs,
                targets = target_sentences, mode = mode, tr_steps=tr_steps)
        elif mode == 'inference':
            sequence_logProb, sequence_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        return sequence_logProb, sequence_predictions

