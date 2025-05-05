import torch
from torch.utils.data import Dataset, DataLoader
import random
from SmilesPE.tokenizer import *
from smallmolec_campaign.utils.smiles_pair_encoders_functions import *
import pandas as  pd

def smilespetok(
            vocab_file = '../../VocabFiles/vocab_spe.txt',
            spe_file = '../../VocabFiles/SPE_ChEMBL.txt'):
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
    return tokenizer

