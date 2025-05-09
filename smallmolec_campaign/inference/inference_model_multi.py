import argparse
import os
import numpy as np
import matplotlib
import pandas as pd
from mpi4py import MPI
import csv
from collections import OrderedDict

matplotlib.use("Agg")
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from tensorflow.keras.preprocessing import sequence, text
#from clr_callback import *
from tensorflow.python.client import device_lib
import json
from ST_funcs.smiles_pair_encoders_functions import *
import time
from ST_funcs.smiles_regress_transformer_funcs import *

def run_inf(data_loc, wght_list):
    comm, size, rank = initialize_mpi()
    
    data_list = sorted(glob.glob(f'data_loc/*gz'))
    data_list_split = np.array_split(data_list, int(size))[rank]
    models = []
    for wght in wght_list:
        
    meta_rank = metadirs[int(rank/48)]
    
    try:
        os.mkdir(f"output_{meta_rank}")
    except:
        pass
    
    model = ModelArchitecture(hyper_params).call()
    model.load_weights(f'{pwd_scren}/Training_multiconfs/DIR.{meta_rank}.oedu/model.weights.3.h5')
    #model.summary()
    
    '''
    Organize data files + setup tokenizer
    '''
    split_files, split_dirs = large_scale_split(hyper_params, 48, rank%48)#size, rank)
    
    if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
        vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
        spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
        tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
    
    ''' 
    Iterate over files
    '''
    
    BATCH = hyper_params['general']['batch_size']
    cutoff = hyper_params['general']['cutoff']
    
    '''
    Inference running on each file
    '''
    
    for fil, dirs in zip(split_files, split_dirs):
    
        Data_smiles_inf, x_inference = large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank)
    
        Output = model.predict(x_inference, batch_size = BATCH)
    
        '''
        Combine SMILES and predicted docking score.
        Sort the data based on the docking score, 
        remove data below cutoff score.
        write data to file in output directory
        '''
        SMILES_DS = np.vstack((Data_smiles_inf, np.array(Output).flatten())).T 
        SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)
    
        filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if item[1] >= cutoff).values())
        filename = f'output_{meta_rank}/{os.path.splitext(fil)[0]}.dat'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smiles', 'score'])
            writer.writerows(filtered_data)
    
        del (Data_smiles_inf)
        del(Output)
        del(x_inference)
        del(SMILES_DS)
        del(filtered_data)
    
    '''
    Sorting all files
    parallel merge sort
    '''
    if True:
        Sorted_data = pd.DataFrame(columns = ['smiles', 'score'])
        
        for fil, dirs in zip(split_files, split_dirs):
            filename = f'output_{meta_rank}/{os.path.splitext(fil)[0]}.dat'
            df = pd.read_csv(filename)
            Sorted_data = pd.concat([Sorted_data, df])
        Sorted_data = Sorted_data.to_numpy()
        Sorted_data = sorted(Sorted_data, key=lambda x: x[1], reverse=True)
        
        Sorted_data = comm.gather(Sorted_data, root=0)
        
        if rank==0:
            print(len(Sorted_data))
            data_to_write = Sorted_data[0]
            for r in range(1,len(Sorted_data)):
                data_to_write.extend(Sorted_data[r])
            data_to_write = sorted(data_to_write, key=lambda x: x[1], reverse=True)
            data_to_write = list(OrderedDict((item[0], item) for item in data_to_write).values())
        
            filename = f'output_{meta_rank}/All.sorted.dat'
        
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['smiles', 'score'])
                writer.writerows(data_to_write[0:10000000])



parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config",
    type=Path,
    required=True,
    help="config file",
)
"""
parser.add_argument(
    "-w", "--weights",
    type=Path,
    required=True,
    help="weights",
)

parser.add_argument(
    "-o", "--output",
    type=Path,
    required=True,
    help="output",
)
"""

args = parser.parse_args()

#try:
#    os.mkdir(args.output)
#except:
#    pass

json_file = args.config
hyper_params = ParamsJson(json_file)

'''
Set up directories to screen
'''
metadirs = ['meta0_pocket1', 'meta1_pocket1', 'meta1_pocket2', 'meta2_pocket1', 'meta2_pocket2', 'meta3_pocket1', 'meta3_pocket2' 'meta3_pocket3']

'''
Load Model + setup mpi
'''

pwd_screen = "/lus/gila/projects/candle_aesp_CNDA/avasan/Workflows/Uchic_Aur_Screens/NMNAT_2"
comm, size, rank = initialize_mpi()
meta_rank = metadirs[int(rank/48)]

try:
    os.mkdir(f"output_{meta_rank}")
except:
    pass

model = ModelArchitecture(hyper_params).call()
model.load_weights(f'{pwd_scren}/Training_multiconfs/DIR.{meta_rank}.oedu/model.weights.3.h5')
#model.summary()

'''
Organize data files + setup tokenizer
'''
split_files, split_dirs = large_scale_split(hyper_params, 48, rank%48)#size, rank)

if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
    spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

''' 
Iterate over files
'''

BATCH = hyper_params['general']['batch_size']
cutoff = hyper_params['general']['cutoff']

'''
Inference running on each file
'''

for fil, dirs in zip(split_files, split_dirs):

    Data_smiles_inf, x_inference = large_inference_data_gen(hyper_params, tokenizer, dirs, fil, rank)

    Output = model.predict(x_inference, batch_size = BATCH)

    '''
    Combine SMILES and predicted docking score.
    Sort the data based on the docking score, 
    remove data below cutoff score.
    write data to file in output directory
    '''
    SMILES_DS = np.vstack((Data_smiles_inf, np.array(Output).flatten())).T 
    SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)

    filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if item[1] >= cutoff).values())
    filename = f'output_{meta_rank}/{os.path.splitext(fil)[0]}.dat'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'score'])
        writer.writerows(filtered_data)

    del (Data_smiles_inf)
    del(Output)
    del(x_inference)
    del(SMILES_DS)
    del(filtered_data)

'''
Sorting all files
parallel merge sort
'''
if True:
    Sorted_data = pd.DataFrame(columns = ['smiles', 'score'])
    
    for fil, dirs in zip(split_files, split_dirs):
        filename = f'output_{meta_rank}/{os.path.splitext(fil)[0]}.dat'
        df = pd.read_csv(filename)
        Sorted_data = pd.concat([Sorted_data, df])
    Sorted_data = Sorted_data.to_numpy()
    Sorted_data = sorted(Sorted_data, key=lambda x: x[1], reverse=True)
    
    Sorted_data = comm.gather(Sorted_data, root=0)
    
    if rank==0:
        print(len(Sorted_data))
        data_to_write = Sorted_data[0]
        for r in range(1,len(Sorted_data)):
            data_to_write.extend(Sorted_data[r])
        data_to_write = sorted(data_to_write, key=lambda x: x[1], reverse=True)
        data_to_write = list(OrderedDict((item[0], item) for item in data_to_write).values())
    
        filename = f'output_{meta_rank}/All.sorted.dat'
    
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smiles', 'score'])
            writer.writerows(data_to_write[0:10000000])
    
