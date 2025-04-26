import glob
import numpy as np
import pandas as pd
import os
import random
import gzip
import shutil
import time
from mpi4py import MPI

def initialize_mpi():
     
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

def split_files(zinc_data_path, num_splits=10):
    # Get all the files in the directory
    all_files = glob.glob(zinc_data_path + '/*.gz')
    
    # Split the files into chunks
    split_files = [all_files[i::num_splits] for i in range(num_splits)]
    
    return split_files

def sample_zinc_random(
        zinc_data_path,
        output_path,
        num_samples=1000,
        seed=42,
        use_mpi=False,
        ):
    if use_mpi:
        comm, rank, size = initialize_mpi()
    else:
        comm = None
        rank = 0
        size = 1
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Split the files into chunks for each MPI process
    # Set random seed for reproducibility   
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #zinc_files = glob.glob(zinc_data_path + '/*.gz')
    files_mpi = split_files(zinc_data_path, num_splits=size)
    # Get the files for this process
    if use_mpi:
        my_files = files_mpi[rank]
    else:
        my_files = files_mpi[0]
    # Sample random SMILES from the files
    # Initialize an empty list to store sampled SMILES
    sampled_smiles = []
    Data_smiles_total = []
    continue_search = 0

    for fil_it, fil in enumerate(my_files):
        with gzip.open(fil,'rt') as f:
                for line in f:
                    Data_smiles_total.append(line.split()[0])

        if len(Data_smiles_total)<100000 and fil_it<len(split_files):
            continue_search=1
            continue
        if continue_search == 0:
                sampled_smiles.append(random.sample(Data_smiles_total, 100))
                Data_smiles_total = []
    return sampled_smiles

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample random SMILES from ZINC dataset.')
    parser.add_argument('--zinc_data_path', type=str, required=True, help='Path to the ZINC dataset directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the sampled SMILES.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to extract from each file.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--use_mpi', action='store_true', help='Use MPI for parallel processing.')
    args = parser.parse_args()

    sampled_smiles = sample_zinc_random(
        zinc_data_path=args.zinc_data_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed,
        use_mpi=args.use_mpi
    )
    # Save the sampled SMILES to a file
    output_file = os.path.join(args.output_path, 'sampled_smiles.txt')
    with open(output_file, 'w') as f:
        for smiles in sampled_smiles:
            f.write(smiles + '\n')
    print(f'Sampled SMILES saved to {output_file}')
    print(f'Sampled {len(sampled_smiles)} SMILES from {args.zinc_data_path}')