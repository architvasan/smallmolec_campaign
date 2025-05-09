import os, sys
import pandas as pd
import scipy as sp
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from SmilesPE.tokenizer import *
from smallmolec_campaign.utils.smiles_pair_encoders_functions import *
from itertools import chain, repeat, islice
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics import BinaryAccuracy
warnings.filterwarnings("ignore")
'''
Initialize tokenizer
'''
vocab_file = './smallmolec_campaign_git/VocabFiles/vocab_spe.txt'
spe_file = 'smallmolec_campaign_git/VocabFiles/SPE_ChEMBL.txt'
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def tokenize_function(examples):
    #print(examples[0])
    return np.array(list(pad(tokenizer(examples)['input_ids'], 64, 0)))

def training_data(raw_data, smilescol, labelcol):
    smiles_data_frame = pd.DataFrame(data = {'text': raw_data[smilescol], 'labels': raw_data[labelcol]})
    #print(smiles_data_frame['text'])
    smiles_data_frame['text'] = smiles_data_frame['text'].map(tokenize_function)#, batched=True)
    #print(smiles_data_frame['text'].values)
    target = smiles_data_frame['labels'].values
    features = np.stack([tok_dat for tok_dat in smiles_data_frame['text']])
    #print(target)
    #train = data_utils.TensorDataset(features, target)
    #train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
    feature_tensor = torch.tensor(features)
    label_tensor = torch.tensor(smiles_data_frame['labels'])
    #print(feature_tensor)
    #print(label_tensor)
    dataset = TensorDataset(feature_tensor, label_tensor)
    #print(len(dataset[0][0]))
    train_size = int(0.9 * len(dataset))
    test_size = int(len(dataset) - train_size)

    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
    #print(training_data.shape)
    #print(len(test_data))
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)
    return train_dataloader, test_dataloader


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        
        self.transformer_encoder1 = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        self.layer_norm = nn.LayerNorm(d_model) 
        self.embedding = nn.Embedding(3132, d_model)
        self.d_model = d_model

        self.dropout1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(6400, 2048)
        self.act1 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(2048, 1024)
        self.act2 = nn.ReLU()

        self.dropout3 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(1024, 256)
        self.act3 = nn.ReLU()

        self.dropout4 = nn.Dropout(0.2)
        self.linear4 = nn.Linear(256, 64)
        self.act4 = nn.Softmax()
        #self.act4 = torch.sigmoid()

        self.dropout5 = nn.Dropout(0.2)
        self.linear5 = nn.Linear(64, 16)
        self.act5 = nn.Softmax()

        self.dropout6 = nn.Dropout(0.2)
        self.linear6 = nn.Linear(16, 1)
        self.act6 = nn.Softmax()



        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear4.bias.data.zero_()
        self.linear4.weight.data.uniform_(-initrange, initrange)
        self.linear5.bias.data.zero_()
        self.linear5.weight.data.uniform_(-initrange, initrange)
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src)* math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder1(src, src_mask)
        output = self.transformer_encoder2(output)
        #output = self.layer_norm(output)
        output = self.dropout1(output)
        output = torch.reshape(output, (len(output),len(output[0])*len(output[0][0])))
        output = self.linear1(output)
        output = self.act1(output)
        output = self.dropout2(output)
        output = self.linear2(output)
        output = self.act2(output)
        output = self.dropout3(output)
        output = self.linear3(output)
        output = self.act3(output)
        output = self.dropout4(output)
        output = self.linear4(output)
        #output = torch.sigmoid(output)
        output = torch.sigmoid(output)
        output = self.dropout5(output)
        output = self.linear5(output)
        output = torch.sigmoid(output)
        output = self.dropout6(output)
        output = self.linear6(output)
        output = torch.sigmoid(output)
        #output = self.act5(output)
        return torch.reshape(output, (-1,))


class TransformerMTLModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        
        self.transformer_encoder1 = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        self.layer_norm = nn.LayerNorm(d_model) 
        self.embedding = nn.Embedding(3132, d_model)
        self.d_model = d_model

        self.dropout1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(512, 256)
        self.act1 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()

        self.dropout3 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(128, 64)
        self.act3 = nn.ReLU()

        self.dropout4 = nn.Dropout(0.2)
        self.linear4 = nn.Linear(64, 32)
        self.act4 = nn.ReLU()
        #self.act4 = torch.sigmoid()

        self.dropout5 = nn.Dropout(0.2)
        self.linear5 = nn.Linear(32, 16)
        self.act5 = nn.ReLU()

        self.dropout6 = nn.Dropout(0.2)
        
        #self.act6 = nn.Softmax()
        # define final layers for each task
        self.final_0 = nn.Linear(16, 1)
        self.final_1 = nn.Linear(16, 1)
        self.final_2 = nn.Linear(16, 1) 
        self.final_3 = nn.Linear(16, 1)
        self.final_4 = nn.Linear(16, 1)
        self.final_5 = nn.Linear(16, 1)
        self.final_6 = nn.Linear(16, 1)
        self.final_7 = nn.Linear(16, 1)
        self.final_8 = nn.Linear(16, 1)
        self.final_9 = nn.Linear(16, 1)
        self.final_10 = nn.Linear(16, 1)
        self.final_11 = nn.Linear(16, 1)
        self.final_12 = nn.Linear(16, 1)
        self.final_13 = nn.Linear(16, 1)
        self.final_14 = nn.Linear(16, 1)
        self.final_15 = nn.Linear(16, 1)
        self.final_16 = nn.Linear(16, 1)
        self.final_17 = nn.Linear(16, 1)

        self.act6 = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear4.bias.data.zero_()
        self.linear4.weight.data.uniform_(-initrange, initrange)
        self.linear5.bias.data.zero_()
        self.linear5.weight.data.uniform_(-initrange, initrange)
        self.final_0.bias.data.zero_()
        self.final_0.weight.data.uniform_(-initrange, initrange)
        self.final_1.bias.data.zero_()
        self.final_1.weight.data.uniform_(-initrange, initrange)
        self.final_2.bias.data.zero_()
        self.final_2.weight.data.uniform_(-initrange, initrange)
        self.final_3.bias.data.zero_()
        self.final_3.weight.data.uniform_(-initrange, initrange)
        self.final_4.bias.data.zero_()
        self.final_4.weight.data.uniform_(-initrange, initrange)
        self.final_5.bias.data.zero_()
        self.final_5.weight.data.uniform_(-initrange, initrange)
        self.final_6.bias.data.zero_()
        self.final_6.weight.data.uniform_(-initrange, initrange)
        self.final_7.bias.data.zero_()
        self.final_7.weight.data.uniform_(-initrange, initrange)
        self.final_8.bias.data.zero_()
        self.final_8.weight.data.uniform_(-initrange, initrange)
        self.final_9.bias.data.zero_()
        self.final_9.weight.data.uniform_(-initrange, initrange)
        self.final_10.bias.data.zero_()
        self.final_10.weight.data.uniform_(-initrange, initrange)
        self.final_11.bias.data.zero_()
        self.final_11.weight.data.uniform_(-initrange, initrange)
        self.final_12.bias.data.zero_()
        self.final_12.weight.data.uniform_(-initrange, initrange)
        self.final_13.bias.data.zero_()
        self.final_13.weight.data.uniform_(-initrange, initrange)
        self.final_14.bias.data.zero_()
        self.final_14.weight.data.uniform_(-initrange, initrange)
        self.final_15.bias.data.zero_()
        self.final_15.weight.data.uniform_(-initrange, initrange)
        self.final_16.bias.data.zero_()
        self.final_16.weight.data.uniform_(-initrange, initrange)
        self.final_17.bias.data.zero_()
        self.final_17.weight.data.uniform_(-initrange, initrange)



    def forward(self, src: Tensor, src_mask: Tensor = None, task_id: int = 0) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src)* math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder1(src, src_mask)
        output = self.transformer_encoder2(output)
        #output = self.layer_norm(output)
        output = self.dropout1(output)
        #output = torch.reshape(output, (len(output),len(output[0])*len(output[0][0])))
        print(output.shape)
        output = self.linear1(output)
        output = self.act1(output)
        output = self.dropout2(output)
        output = self.linear2(output)
        output = self.act2(output)
        output = self.dropout3(output)
        output = self.linear3(output)
        output = self.act3(output)
        output = self.dropout4(output)
        output = self.linear4(output)
        #output = torch.sigmoid(output)
        #output = torch.sigmoid(output)
        output = self.act4(output)
        output = self.dropout5(output)
        output = self.linear5(output)
        output = self.act5(output)
        output = self.dropout6(output)
        if task_id == 0:
            output = self.final_0(output)
        elif task_id == 1:
            output = self.final_1(output)
        elif task_id == 2:
            output = self.final_2(output)
        elif task_id == 3:
            output = self.final_3(output)
        elif task_id == 4:
            output = self.final_4(output)
        elif task_id == 5:
            output = self.final_5(output)
        elif task_id == 6:
            output = self.final_6(output)
        elif task_id == 7:
            output = self.final_7(output)
        elif task_id == 8:
            output = self.final_8(output)
        elif task_id == 9:
            output = self.final_9(output)
        elif task_id == 10:
            output = self.final_10(output)
        elif task_id == 11:
            output = self.final_11(output)
        elif task_id == 12:
            output = self.final_12(output)
        elif task_id == 13:
            output = self.final_13(output)
        elif task_id == 14:
            output = self.final_14(output)
        elif task_id == 15:
            output = self.final_15(output)
        elif task_id == 16:
            output = self.final_16(output)
        elif task_id == 17:
            output = self.final_17(output)
        else:
            assert False, 'Bad Task ID passed'
        #output = self.linear6(output)
        output = self.act6(output)
        #output = torch.(output)
        #output = self.act5(output)
        return output#torch.reshape(output, (-1,))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#directory = 'data'
#dir_list = os.listdir(directory)

model = TransformerMTLModel(ntoken=64,
                            d_model = 512,
                            nhead = 16,
                            d_hid = 128,
                            nlayers = 16,
                            dropout = 0.1)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
loss_fn = nn.MSELoss()

'''
Multi task learning tests
'''

df = pd.read_csv('/eagle/FoundEpidem/avasan/IDEAL/SmallMolecCampaign/whsc1/data_training.csv')

train_dataloaders = []
test_dataloaders = []
for col in df.columns:
    if col == 'smiles':
        continue
    else:
        dls = training_data(df, 'smiles', col)
        trn_dl = dls[0]
        tst_dl = dls[1]
        train_dataloaders.append(trn_dl)
        test_dataloaders.append(tst_dl)

import itertools
from itertools import zip_longest
for ep in range(6):
    for batches in zip_longest(*train_dataloaders, fillvalue=None):
        #loss = 0
        for i, (batchX, batchY) in enumerate(batches):
            batchX = batchX.to(device)
            batchY = batchY.to(device)
            #print(batchX)
            pred_i = model(batchX, task_id = i)
            if i == 0:
                loss=loss_fn(pred_i, batchY)
            else:
                loss+=loss_fn(pred_i, batchY)
            #del(pred_i)
            if batchX is not None:
                #print(batchX)
                print(batchY)
                print(pred_i)
        pred_i = pred_i.float()
        print(pred_i.shape)
        batchY = batchY.float()
        print(batchY.shape)
        loss = loss.float()
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #for j, ((batch_X, batch_Y)) in enumerate(dataloaders[0]):
    #    print(batch_X)
    #    print(batch_Y)
    #for item in itertools.chain(*dataloaders):
    #    print(item[0])


import sys
sys.exit()
for i in range(6):
    zipped_dls = zip(movie_dl, yelp_dl)
    for j, ((movie_batch_X, movie_batch_y), (yelp_batch_X, yelp_batch_y)) in enumerate(zipped_dls):
        
        movie_preds = model(movie_batch_X, task_id = 0)
        movie_loss = movie_loss_fn(movie_preds, movie_batch_y)
        
        yelp_preds = model(yelp_batch_X, task_id = 1)
        yelp_loss = yelp_loss_fn(yelp_preds, yelp_batch_y)
        
        loss = movie_loss + yelp_loss
        losses_per_epoch.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
