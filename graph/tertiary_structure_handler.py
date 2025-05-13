import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.esmfold import esmfold_model_handler as esmfold
from utils import pdb_parser, distances

def load_tertiary_structures(sequences,pdb_path):

    if not pdb_path.exists():
        logging.warning(f"PDB directory {pdb_path} does not exist. Cannot load structures.")
        # Return a list of Nones or empty arrays, matching the expected output structure
        return [np.array([], dtype='float64') for _ in sequences] 

    sequences_to_exclude = pd.DataFrame()
    atom_coordinates_matrices = []
    with tqdm(range(len(sequences)), total=len(sequences), desc="Loading pdb files", disable=False) as progress:
        pdbs = []
        for row in sequences:
            pdb_file = pdb_path.joinpath(f"{row}.pdb")
            try:
                pdb_str = pdb_parser.open_pdb(pdb_file)
                pdbs.append(pdb_str)
                coordinates_matrix = np.array(pdb_parser.get_atom_coordinates_from_pdb(pdb_str,'CA'),
                             dtype='float64')
                coordinates_matrix = np.array(distances.translate_positive_coordinates(coordinates_matrix), dtype='float64')
                atom_coordinates_matrices.append(coordinates_matrix)
                progress.update(1)
            except Exception as e:
                sequences_to_exclude = sequences_to_exclude.append(row)

        return atom_coordinates_matrices


def predict_tertiary_structures(sequences,pdb_path):
    
    # Ensure the output directory for PDBs exists
    if not pdb_path.exists():
        pdb_path.mkdir(parents=True)

    pdbs = esmfold.predict_structures(sequences) # This now returns a list of PDB strings
    pdb_names = [str(row) for row in sequences]

    atom_coordinates_matrices = []
    # tqdm description updated
    with tqdm(range(len(pdbs)), total=len(pdbs), desc="Saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            
            pdb_parser.save_pdb(pdb_str, pdb_name, pdb_path)
            
            coordinates_matrix = np.array(pdb_parser.get_atom_coordinates_from_pdb(pdb_str, 'CA'),
                         dtype='float64')
            coordinates_matrix = np.array(distances.translate_positive_coordinates(coordinates_matrix), dtype='float64')
            atom_coordinates_matrices.append(coordinates_matrix)
            progress.update(1)            

    logging.info(f"Predicted tertiary structures saved in: {pdb_path.resolve()}")
    return atom_coordinates_matrices


