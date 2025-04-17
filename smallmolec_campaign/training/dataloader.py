import os, sys
from torchsummary import summary
import pandas as pd
import scipy as sp
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import movie_reviews
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from SmilesPE.tokenizer import *
from smiles_pair_encoders_functions import *
from itertools import chain, repeat, islice
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics import BinaryAccuracy
warnings.filterwarnings("ignore")
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import json
from torcheval.metrics import R2Score
from tqdm import tqdm
import wandb

'''
Initialize tokenizer
'''
vocab_file = 'VocabFiles/vocab_spe.txt'
spe_file = 'VocabFiles/SPE_ChEMBL.txt'
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def tokenize_function(examples, ntoken, tokenizer):
    return np.array(list(pad(tokenizer(examples)['input_ids'], ntoken, 0)))

def training_data(raw_data, smiles_col, labels_col):
    smiles_data_frame = pd.DataFrame(data = {'text': raw_data[smiles_col], 'labels': raw_data[labels_col]})
    smiles_data_frame['text'] = smiles_data_frame['text'].apply(lambda x: tokenize_function(x, ntoken=ntoken))#map(tokenize_function)#, batched=True)
    target = smiles_data_frame['labels'].values
    features = np.stack([tok_dat for tok_dat in smiles_data_frame['text']])
    feature_tensor = torch.tensor(features)
    label_tensor = torch.tensor(smiles_data_frame['labels'])
    dataset = TensorDataset(feature_tensor, label_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)

    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
    return train_dataloader, test_dataloader, test_data


def dataload_presplit(traindat, valdat, smilescol, labelcol, batch, ntoken):
    tqdm.pandas()
    smiles_df_train = pd.DataFrame(data = {'text': traindat[smilescol], 'labels': traindat[labelcol]})
    smiles_df_val = pd.DataFrame(data = {'text': valdat[smilescol], 'labels': valdat[labelcol]})

    smiles_df_train['text'] = smiles_df_train['text'].progress_apply(lambda x: tokenize_function(x, ntoken=ntoken))
    target_train = smiles_df_train['labels'].values
    features_train = [tok_dat for tok_dat in smiles_df_train['text']]#np.stack([tok_dat for tok_dat in smiles_df_train['text']])
    smiles_df_val['text'] = smiles_df_val['text'].progress_apply(lambda x: tokenize_function(x, ntoken=ntoken))
    target_val = smiles_df_val['labels'].values
    features_val = [tok_dat for tok_dat in smiles_df_val['text']]#np.stack([tok_dat for tok_dat in smiles_df_val['text']])

    feature_tensor_train = torch.tensor(features_train)
    label_tensor_train = torch.tensor(smiles_df_train['labels'])
    feature_tensor_val = torch.tensor(features_val)
    label_tensor_val = torch.tensor(smiles_df_val['labels'])

    train_dataset = TensorDataset(feature_tensor_train, label_tensor_train)
    val_dataset = TensorDataset(feature_tensor_val, label_tensor_val)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    return train_dataloader, val_dataloader, val_dataset

