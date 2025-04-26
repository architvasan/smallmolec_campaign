"""This module contains functions docking a molecule to a receptor using Openeye.

The code is adapted from this repository: https://github.com/inspiremd/Model-generation
"""
#import MDAnalysis as mda
import sys
import glob
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path
from typing import List, Optional
import numpy as np
from openeye import oechem, oedocking, oeomega
import pandas as pd
from mpi4py import MPI
from tqdm import tqdm
from smallmolec_campaign.htp_docking.docking_utils import smi_to_structure
from smallmolec_campaign.htp_docking.utils import exception_handler
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from natsort import natsorted
'''
Functions
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import sample
import pickle
from mpi4py import MPI
import random
import MDAnalysis as mda
from dataclasses import dataclass

random_state = 1

def init_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size

def sample_dataset(data_pwd, dataset, size, rank, sample_size):
    list_dir_files = np.array(sorted(os.listdir(f'{data_pwd}/{dataset}')))
    num_files = len(list_dir_files)
    list_dir_files = np.array_split(list_dir_files, size)[rank]
    n = int(sample_size/num_files)+1#sample_size_per_file
    sample_dataset = []#np.empty(sample_size, dtype='str')#[]
    sample_perrank = sample_size/size
    for fil in list_dir_files:
        with open(f'{data_pwd}/{dataset}/{fil}', 'r') as fil:
            data = fil.read().splitlines()[1:]
            del(fil)
        sample_fil = sample(data, n)
        sample_dataset.extend(sample_fil)
        del(sample_fil)
        del(data)
        if len(sample_dataset)>=sample_perrank:
            return sample_dataset
    return sample_dataset

@dataclass
class BaseSettings:
    """ Loading config file with params:
        recep_file_dir : Path
            Path to the receptor .oedu file.
        max_confs : int
            Number of ligand poses to generate
        temp_lig_dir : Path
            Temporary directory to write individual ligand poses
        out_lig_path : Path
            Location to output ligand_protein top n poses --> single pdb
        out_rec_path : Path
            Where to write output receptor file
        temp_storage : Path
            Path to the temporary storage directory to write structures to,
            if None, use the current working Python's built in temp storage.
    """
    config_fil: str

    def __post_init__(self):
        ### Parameters setting
        import json
         
        # Opening JSON file
        f = open(self.config_fil)
         
        # returns JSON object as 
        # a dictionary
        config = json.load(f)
        
        self.data_smi = config['smiles_input']
        self.recep_file_dir = f'{config["recep_loc"]}' # receptor oedu file for openeye 
        self.score_dir = config['score_directory']
        self.score_pattern_base = config['score_pattern']#'scores/4ui5_scores' #store scores like this (will have #ranks files)
        self.protein_pdb = config['protein_pdb'] #protein pdb file to use to store. Will save everything in this file to complex
        self.max_confs = config['max_confs'] # confs to generate
        self.pose_gen = config['pose_gen'] # Boolean... should we gen poses?
        self.batch_size = config['batch_size']
        self.score_cutoff = config['score_cutoff'] # below this score generate pose
        self.temp_dir = config['temp_storage']
        self.out_poses_gen = config['out_poses']#'lig_confs' # store ligand poses temporarily

        # Check whether the specified path exists or not
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.score_dir, exist_ok = True)
        self.recep_files = natsorted(glob.glob(f"{self.recep_file_dir}/*.oedu"))
        self.sample_size = 0#config['sample_size'] 
        
        ### Initialize mpi
        self.comm, self.rank, self.size = init_mpi()
        
class OpeneyeFuncs():
    def __init__(self, config_fil):
        return
    @staticmethod
    def smi_to_structure(smiles: str, output_file: Path, forcefield: str = "mmff") -> None:
        """Convert a SMILES file to a structure file.
    
        Parameters
        ----------
        smiles : str
            Input SMILES string.
        output_file : Path
            EIther an output PDB file or output SDF file.
        forcefield : str, optional
            Forcefield to use for 3D conformation generation
            (either "mmff" or "etkdg"), by default "mmff".
        """
        # Convert SMILES to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
    
        # Add hydrogens to the molecule
        mol = Chem.AddHs(mol)
    
        # Generate a 3D conformation for the molecule
        if forcefield == "mmff":
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
        elif forcefield == "etkdg":
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        else:
            raise ValueError(f"Unknown forcefield: {forcefield}")
    
        # Write the molecule to a file
        if output_file.suffix == ".pdb":
            writer = Chem.PDBWriter(str(output_file))
        elif output_file.suffix == ".sdf":
            writer = Chem.SDWriter(str(output_file))
        else:
            raise ValueError(f"Invalid output file extension: {output_file}")
        writer.write(mol)
        writer.close()
    
    @staticmethod
    def from_mol(mol, isomer=True, num_enantiomers=1):
        """Generates a set of conformers as an OEMol object
        Inputs:
            mol is an OEMol
            isomers is a boolean controling whether or not the various diasteriomers of a molecule are created
            num_enantiomers is the allowable number of enantiomers. For all, set to -1
        """
        # Turn off the GPU for omega
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.GetTorDriveOptions().SetUseGPU(False)
        omega = oeomega.OEOmega(omegaOpts)
    
        out_conf = []
        if not isomer:
            ret_code = omega.Build(mol)
            if ret_code == oeomega.OEOmegaReturnCode_Success:
                out_conf.append(mol)
            else:
                oechem.OEThrow.Warning(
                    "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                )
    
        elif isomer:
            for enantiomer in oeomega.OEFlipper(mol.GetActive(), 12, True):
                enantiomer = oechem.OEMol(enantiomer)
                ret_code = omega.Build(enantiomer)
                if ret_code == oeomega.OEOmegaReturnCode_Success:
                    out_conf.append(enantiomer)
                    num_enantiomers -= 1
                    if num_enantiomers == 0:
                        break
                else:
                    oechem.OEThrow.Warning(
                        "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                    )
        return out_conf
    
    @staticmethod
    def from_string(smiles, isomer=True, num_enantiomers=1):
        """
        Generates an set of conformers from a SMILES string
        """
        mol = oechem.OEMol()
        if not oechem.OESmilesToMol(mol, smiles):
            raise ValueError(f"SMILES invalid for string {smiles}")
        else:
            return OpeneyeFuncs.from_mol(mol, isomer, num_enantiomers)
    
    @staticmethod
    def from_structure(structure_file: Path) -> oechem.OEMol:
        """
        Generates an set of conformers from a SMILES string
        """
        mol = oechem.OEMol()
        ifs = oechem.oemolistream()
        if not ifs.open(str(structure_file)):
            raise ValueError(f"Could not open structure file: {structure_file}")
    
        if structure_file.suffix == ".pdb":
            oechem.OEReadPDBFile(ifs, mol)
        elif structure_file.suffix == ".sdf":
            oechem.OEReadMDLFile(ifs, mol)
        else:
            raise ValueError(f"Invalid structure file extension: {structure_file}")
    
        return mol
    
    @staticmethod
    def select_enantiomer(mol_list):
        return mol_list[0]

    @staticmethod
    def init_rec(receptor):
        dock = oedocking.OEDock()
        dock.Initialize(receptor)
        return dock
    
    @staticmethod
    def dock_conf(dock, mol, max_poses: int = 1):
        lig = oechem.OEMol()
        err = dock.DockMultiConformerMolecule(lig, mol, max_poses)
        return dock, lig
    
    @staticmethod
    # Returns an array of length max_poses from above. This is the range of scores
    def ligand_scores(dock, lig):
        return [dock.ScoreLigand(conf) for conf in lig.GetConfs()]
    
    @staticmethod
    def write_ligand(ligand, output_dir: Path, smiles: str, lig_identify: str) -> None:
        # TODO: If MAX_POSES != 1, we should select the top pose to save
        ofs = oechem.oemolostream()
        for it, conf in enumerate(list(ligand.GetConfs())):
            if ofs.open(f'{str(output_dir)}/{lig_identify}/{it}.pdb'):
                oechem.OEWriteMolecule(ofs, conf)
                ofs.close()
        return
        raise ValueError(f"Could not write ligand to {output_path}")
    
    @staticmethod
    def write_receptor(receptor, output_path: Path) -> None:
        ofs = oechem.oemolostream()
        if ofs.open(str(output_path)):
            mol = oechem.OEMol()
            contents = receptor.GetComponents(mol)#Within
            oechem.OEWriteMolecule(ofs, mol)
            ofs.close()
        return
        raise ValueError(f"Could not write receptor to {output_path}")
    
    @staticmethod
    @cache  # Only read the receptor once
    def read_receptor(receptor_oedu_file: Path):
        """Read the .oedu file into a GraphMol object."""
        receptor = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(receptor_oedu_file), receptor)
        return receptor
    
    @staticmethod
    @cache
    def create_proteinuniv(protein_pdb):
        protein_universe = mda.Universe(protein_pdb)
        return protein_universe
    
    @staticmethod
    def create_complex(protein_universe, ligand_pdb):
        u1 = protein_universe
        u2 = mda.Universe(ligand_pdb)
        u = mda.core.universe.Merge(u1.select_atoms("all"), u2.atoms)#, u3.atoms)
        return u

    @staticmethod
    def create_trajectory(protein_universe, ligand_dir, output_pdb_name, output_dcd_name):
        import MDAnalysis as mda
        ligand_files = sorted(os.listdir(ligand_dir))
        comb_univ_1 = OpeneyeFuncs.create_complex(protein_universe, f'{ligand_dir}/{ligand_files[0]}').select_atoms("all")
    
        with mda.Writer(output_pdb_name, comb_univ_1.n_atoms) as w:
            w.write(comb_univ_1)
        with mda.Writer(output_dcd_name, comb_univ_1.n_atoms,) as w:
            for it, ligand_file in enumerate(ligand_files):
                comb_univ = OpeneyeFuncs.create_complex(protein_universe, f'{ligand_dir}/{ligand_file}') 
                w.write(comb_univ)    # write a whole universe
                os.remove(f'{ligand_dir}/{ligand_file}')
        return
    
class RunDocking(BaseSettings, OpeneyeFuncs):
    def __init__(self, config_fil):
        super().__init__(config_fil)

        self.smi_df = pd.read_csv(self.data_smi)
        self.smi_df_rank =  np.array_split(self.smi_df, self.size)[self.rank] 
        self.receptor_inps = [self.read_receptor(rec) for rec in self.recep_files]
        self.dock_objs = [self.init_rec(rec_inp) for rec_inp in self.receptor_inps]
        self.score_dict = {os.path.splitext(os.path.basename(rec))[0]: [] for rec in self.recep_files}
        self.score_dict['smiles'] = []
        if self.pose_gen == True:
            for rec_file, rec_inp in zip(self.recep_files, self.receptor_inps):
                out_rec_path = f'{self.temp_dir}/{Path(rec_file).stem}.pdb'
                if not os.path.isfile(out_rec_path):
                    self.write_receptor(rec_inp, out_rec_path)
    

    @exception_handler(default_return=0.0)
    def dock_smile_rec(self,
                       smiles,
                       conformers,
                       recf,
                       dock_obj,
                       lig_identify = None,
                       output_poses = None):
        """Run OpenEye docking on a single ligand, receptor pair.
        Parameters
        ----------
        smiles : ste
            A single SMILES string.
        """
        # Dock the ligand conformers to the receptor
        dock, lig = self.dock_conf(dock_obj, conformers, max_poses = self.max_confs)
    
        # Get the docking scores
        best_score = self.ligand_scores(dock, lig)[0]
    
        if self.pose_gen:
            if best_score[0]<self.score_cutoff:
                sys.stdout.flush()
                if not os.path.isdir(f'{self.out_poses_gen}/{lig_identify}'):
                    os.mkdir(f'{self.out_poses_gen}/{lig_identify}', exist_ok = True)
                self.write_ligand(lig, f'{self.out_poses_gen}', smiles, lig_identify)
                protein_universe = self.create_proteinuniv(self.protein_pdb)
                
                os.makedirs(f'{self.out_poses_gen}/{lig_identify}/pdbs')
                os.makedirs(f'{self.out_poses_gen}/{lig_identify}/dcds')
                self.create_trajectory(protein_universe, f'{self.out_poses_gen}/{lig_identify}',
                                        f'{self.out_poses_gen}/{lig_identify}/pdbs/{Path(recf).stem}.{lig_identify}.pdb',
                                        f'{self.out_poses_gen}/{lig_identify}/dcds/{Path(recf).stem}.{lig_identify}.dcd')

        return best_score

    def run_docking(
        self,
        smiles: str, 
        lig_identify: str | None = None,
    ) -> float:
        """Run OpenEye docking on a single ligand and multiple receptors.
        Parameters
        ----------
        smiles : ste
            A single SMILES string.
        """
    
        try:
            conformers = self.select_enantiomer(self.from_string(smiles))
        except:
            with tempfile.NamedTemporaryFile(suffix=".pdb", dir=self.temp_dir) as fd:
                # Read input SMILES and generate conformer
                self.smi_to_structure(smiles, Path(fd.name))
                conformers = self.from_structure(Path(fd.name))
        self.score_dict['smiles'].append(smiles)
        for recit, (recf, dock_obj) in enumerate(zip(self.recep_files, self.dock_objs)):
            recf_base = os.path.splitext(os.path.basename(recf))[0]
            score = self.dock_smile_rec(
                       smiles,
                       conformers,
                       recf,
                       dock_obj,
                       lig_identify = lig_identify)
            print(score)
            self.score_dict[recf_base].append(score)
    
    def run_mpi_docking_list(self,
                             lig_identify = None):
        
        smiles_list = self.smi_df_rank['smiles']
        smiles_index = [ind for ind in range(len(smiles_list))]
        smiles_list_batches = np.array_split(smiles_list, int(len(smiles_list)//self.batch_size))
        smiles_index_batches = np.array_split(smiles_index, int(len(smiles_index)//self.batch_size))
        for batch_it, (batch_s, batch_i) in tqdm(enumerate(zip(smiles_list_batches, smiles_index_batches))):
            for smi, smi_ind in zip(batch_s, batch_i ):
                if self.pose_gen:
                    lig_identify = smi_ind
                self.run_docking(
                    smi, 
                    lig_identify,
                    )
            df_score = pd.DataFrame(self.score_dict)
            df_score.to_csv(f'{self.score_dir}/{self.score_pattern_base}.{self.rank}.{batch_it}.csv')
            self.score_dict = {os.path.splitext(os.path.basename(rec))[0]: [] for rec in self.recep_files}
            self.score_dict['smiles'] = []
        return 1

if __name__ == "__main__":
    '''
    Running Code
    '''
    
    import argparse
    
    
    parser = argparse.ArgumentParser(description='load config file')
    parser.add_argument("-c", "--config", type=Path
    )
    args = parser.parse_args()
    
    run_docking = RunDocking(args.config)
    run_docking.run_mpi_docking_list()

    

