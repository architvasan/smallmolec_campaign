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

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

vocab_file = '../../VocabFiles/vocab_spe.txt'
spe_file = '../../VocabFiles/SPE_ChEMBL.txt'
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

data = pd.read_csv('../../data/sample.train')
data_smi = list(data['smiles'])

tok_data = tokenizer.encode_plus(data_smi[0], 
                                 add_special_tokens=True,
                                 max_length=64,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors='pt')
input_ids = tok_data['input_ids'].squeeze()
attention_mask = tok_data['attention_mask'].squeeze()
print(tok_data)



