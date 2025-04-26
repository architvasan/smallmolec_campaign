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
from smallmolec_campaign.training.smiles_pair_encoders_functions import *
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

#'''
#Initialize tokenizer
#'''
#vocab_file = 'VocabFiles/vocab_spe.txt'
#spe_file = 'VocabFiles/SPE_ChEMBL.txt'
#tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

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

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len=64):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data)
        self.lines = data

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
         # Adding PAD token for labels
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label

    def get_sent(self, index):
        '''return random sentence pair'''
        t1, t2 = self.get_corpus_line(index)

        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        '''return random single sentence'''
        return self.lines[random.randrange(len(self.lines))][1]