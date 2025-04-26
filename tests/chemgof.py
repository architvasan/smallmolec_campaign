from chemicalgof import Smiles2GoF
from chemicalgof import CanonicalGoF2Tokens
import pandas as pd
from tqdm import tqdm

def smi2fsmi(smiles):
    ## Example SMILES string of a molecule
    #smiles = 'C[C@@](O)(Cl)C(=O)NC[C@@H]1CC[C@H](C(=O)O)O1' ## molecule provides chirality information
    
    ## Convert SMILES to directed graph !
    DiG = Smiles2GoF(smiles)
    
    T=CanonicalGoF2Tokens(DiG)
    
    ### then get sequence of tokens for fragments and bonds
    fragsmi = T.getSequence()
    
    ## or simply each fragment and its bonds splitted by dots
    fragsmi = T.getString()
    
    return fragsmi

df = pd.read_csv('../data/sample.train')

df_smil = list(df['smiles'])
for smi in tqdm(df_smil):
    try:
        fsmi = smi2fsmi(smi)
        #print(smi2fsmi(smi))
    except Exception as e:
        print(e)
        print(smi)
