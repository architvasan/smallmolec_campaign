import glob
import numpy as np
import pandas as pd
import os
import random
import gzip
import shutil
import time
 
def sample_zinc_random(
        zinc_data_path,
        output_path,
        num_samples=1000,
        seed=42,
        ):
    
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    zinc_files = glob.glob(zinc_data_path + '/*.gz')
    sampled_smiles = []
    Data_smiles_total = []
    continue_search = 0

    for fil_it, fil in enumerate(zinc_files):
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

    args = parser.parse_args()

    sampled_smiles = sample_zinc_random(
        zinc_data_path=args.zinc_data_path,
        output_path=args.output_path,
        num_samples=args.num_samples
    )
    # Save the sampled SMILES to a file
    output_file = os.path.join(args.output_path, 'sampled_smiles.txt')
    with open(output_file, 'w') as f:
        for smiles in sampled_smiles:
            f.write(smiles + '\n')
    print(f'Sampled SMILES saved to {output_file}')
    print(f'Sampled {len(sampled_smiles)} SMILES from {args.zinc_data_path}')