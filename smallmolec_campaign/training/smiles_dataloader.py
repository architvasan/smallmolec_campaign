import torch
from torch.utils.data import Dataset, DataLoader
import random
from SmilesPE.tokenizer import *
from smallmolec_campaign.training.smiles_pair_encoders_functions import *
import pandas as  pd

def smilespetok(
            vocab_file = '../../VocabFiles/vocab_spe.txt',
            spe_file = '../../VocabFiles/SPE_ChEMBL.txt'):
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
    return tokenizer

class MaskedLanguageModelingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        # Create labels and mask some tokens
        labels = input_ids.clone()
        mask_indices = (torch.rand(input_ids.shape) < self.mask_prob) &\
                       (input_ids != self.tokenizer.pad_token_id) &\
                       (input_ids != self.tokenizer.cls_token_id) &\
                       (input_ids != self.tokenizer.sep_token_id)

        input_ids[mask_indices] = self.tokenizer.mask_token_id

        return input_ids, attention_mask, labels

class MLMDataloader:
    def __init__(self, texts, tokenizer, batch_size=32, max_length=64, mask_prob=0.15, shuffle=True):
        self.dataset = MaskedLanguageModelingDataset(texts, tokenizer, max_length, mask_prob)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

if __name__ == "__main__":
    vocab_file = '../../VocabFiles/vocab_spe.txt'
    spe_file = '../../VocabFiles/SPE_ChEMBL.txt'
    tokenizer = smilespetok(vocab_file=vocab_file, spe_file= spe_file)

    data = pd.read_csv('../../data/sample.train')
    texts = list(data['smiles'])
    # Example usage:
    # Assuming you have a custom tokenizer object with the necessary methods
    mlm_dataloader = MLMDataloader(texts, tokenizer)
    dataloader = mlm_dataloader.get_dataloader()

    #Iterate through the dataloader
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        # Use these tensors for training your model
        #print(input_ids)
        #print(attention_mask)
        #print(labels)