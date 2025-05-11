import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.esmfold import esmfold_model_handler as esmfold
from utils import pdb_parser, distances, get_atom_coordinates_from_pdb, _get_random_coordinates


def load_tertiary_structures(sequences,pdb_path):

    if not pdb_path.exists():
        logging.warning(f"PDB directory {pdb_path} does not exist. Cannot load structures.")
        # Return a list of Nones or empty arrays, matching the expected output structure
        return [np.array([], dtype='float64') for _ in sequences] 

    # sequences_to_exclude = pd.DataFrame() # This was for appending rows, better to collect list of indices/sequences
    excluded_sequences_info = []
    atom_coordinates_matrices = []
    
    # Assuming sequences is a list of sequence strings or identifiers that map to PDB filenames
    with tqdm(enumerate(sequences), total=len(sequences), desc="Loading PDB files and extracting coordinates") as progress_bar:
        for i, seq_identifier in progress_bar:
            # Adapt filename generation to how PDBs were saved by predict_tertiary_structures
            pdb_file_name = f"{str(seq_identifier)}.pdb"
            pdb_file = pdb_path.joinpath(pdb_file_name)

            try:
                if not pdb_file.exists():
                    # logging.warning(f"PDB file not found: {pdb_file}. For sequence: {seq_identifier}")
                    alt_pdb_name_prefix = str(seq_identifier)[:30] + "..." if len(str(seq_identifier)) > 30 else str(seq_identifier)
                    alt_pdb_name = f"seq_{i}_{alt_pdb_name_prefix.replace('/', '_').replace(' ', '_')}.pdb"
                    pdb_file = pdb_path.joinpath(alt_pdb_name)
                    if not pdb_file.exists():
                        logging.warning(f"PDB file still not found: {pdb_file} (tried original and alt name). For sequence: {seq_identifier}")
                        excluded_sequences_info.append({'index': i, 'identifier': seq_identifier, 'reason': 'PDB file not found'})
                        atom_coordinates_matrices.append(np.array([], dtype='float64')) # Placeholder for missing
                        continue

                pdb_str = open_pdb(pdb_file)
                coordinates_list = get_atom_coordinates_from_pdb(pdb_str, 'CA')
                
                if not coordinates_list:
                    logging.warning(f"No CA coordinates found in {pdb_file} for sequence: {seq_identifier}")
                    excluded_sequences_info.append({'index': i, 'identifier': seq_identifier, 'reason': 'No CA coordinates'})
                    atom_coordinates_matrices.append(np.array([], dtype='float64'))
                    continue

                coordinates_matrix = np.array(coordinates_list, dtype='float64')
                
                if coordinates_matrix.size == 0:
                    logging.warning(f"Empty coordinate matrix from {pdb_file} for sequence: {seq_identifier}")
                    atom_coordinates_matrices.append(np.array([], dtype='float64'))
                    continue

                translated_coordinates = translate_positive_coordinates(coordinates_matrix.tolist())
                atom_coordinates_matrices.append(np.array(translated_coordinates, dtype='float64'))
            
            except Exception as e:
                logging.error(f"Error processing PDB for sequence {seq_identifier} (file: {pdb_file}): {e}")
                excluded_sequences_info.append({'index': i, 'identifier': seq_identifier, 'reason': str(e)})
                atom_coordinates_matrices.append(np.array([], dtype='float64')) # Placeholder on error

    if excluded_sequences_info:
        logging.warning(f"Excluded {len(excluded_sequences_info)} sequences during PDB loading. Details: {excluded_sequences_info}")

    return atom_coordinates_matrices


def predict_tertiary_structures(sequences,pdb_path):
    
    # Ensure the output directory for PDBs exists
    if not pdb_path.exists():
        pdb_path.mkdir(parents=True)

    pdbs = predict_structures(sequences) # This now returns a list of PDB strings
    pdb_names = [str(seq)[:30] + "..." if len(str(seq)) > 30 else str(seq) for seq in sequences] # Example naming

    atom_coordinates_matrices = []
    # tqdm description updated
    with tqdm(zip(pdb_names, pdbs, sequences), total=len(pdbs), desc="Processing PDBs & extracting coordinates") as progress_bar:
        for i, (pdb_name_prefix, pdb_str, original_sequence) in enumerate(progress_bar):
            # Create a more unique PDB name, e.g., using an index or a hash of the sequence
            unique_pdb_name = f"seq_{i}_{pdb_name_prefix.replace('/', '_').replace(' ', '_')}" # Make name file-system friendly
            save_pdb(pdb_str, unique_pdb_name, pdb_path)
            
            coordinates_list = get_atom_coordinates_from_pdb(pdb_str, 'CA')
            if not coordinates_list:
                # Handle cases where no CA atoms are found or PDB is problematic
                # Option 1: Log a warning and append None or an empty array
                logging.warning(f"No CA coordinates found for sequence {i}: {original_sequence[:30]}... Skipping.")
                atom_coordinates_matrices.append(np.array([], dtype='float64')) # Or None
                continue 
                # Option 2: Raise an error, depending on how critical this is
                # raise ValueError(f"Failed to get CA coordinates for sequence {i}")

            coordinates_matrix = np.array(coordinates_list, dtype='float64')
            
            # Check if coordinates_matrix is empty before translating
            if coordinates_matrix.size == 0:
                 logging.warning(f"Empty coordinate matrix for sequence {i} before translation. PDB: {unique_pdb_name}")
                 atom_coordinates_matrices.append(np.array([], dtype='float64'))
                 continue

            translated_coordinates = translate_positive_coordinates(coordinates_matrix.tolist()) # tolist() if needed by translate
            atom_coordinates_matrices.append(np.array(translated_coordinates, dtype='float64'))
            
    # Logging info about where PDBs are saved
    # workflow_logger might not be defined here. Using standard logging.
    logging.info(f"Predicted tertiary structures saved in: {pdb_path.resolve()}")
    return atom_coordinates_matrices


