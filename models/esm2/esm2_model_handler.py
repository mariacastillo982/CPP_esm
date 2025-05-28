import numpy as np
import pandas as pd
import torch
from torch import hub
import esm
from esm import FastaBatchedDataset
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from tqdm import tqdm
from utils import json_parser
import collections
from pathlib import Path
import gc
torch.cuda.empty_cache()

def get_models(esm2_representation):
    """
    :param esm2_representation: residual-level features representation name
    :return:
        models: models corresponding to the specified esm 2 representation
    """

    esm2_representations_json = os.getcwd() + os.sep + "settings/esm2_representations.json"
    data = json_parser.load_json(esm2_representations_json)

    # Create a DataFrame
    representations = pd.DataFrame(data["representations"])

    # Filter encoding method
    representation = representations[representations["representation"] == esm2_representation]

    # Check if the DataFrame is empty
    if not representation.empty:
        # Extract the column "models" and create a new DataFrame
        models = representation["models"].explode(ignore_index=True)
    else:
        #  If the DataFrame is empty, throw an exception and stop the code.
        raise Exception(f"'{esm2_representation}' is not a valid coding method name.")

    return models


def get_embeddings(data, model_name, reduced_features, validation_mode, randomness_percentage, use_esm2_contact_map):
    """
    get_embeddings
    :param use_esm2_contact_map:
    :param randomness_percentage:
    :param validation_mode:
    :param ids: sequences identifiers. Containing multiple sequences.
    :param sequences: sequences itself
    :param model_name: esm2 model name
    :param reduced_features: vector of positions of the features to be used
    :return:
        embeddings: reduced embedding of each sequence of the fasta file according to reduced_features
    """
    try:
        # esm2 checkpoints
        hub.set_dir(os.getcwd() + os.sep + "models/esm2/")

        no_gpu = False
        model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # disables dropout for deterministic results

        if torch.cuda.is_available() and not no_gpu:
            model = model.cuda()
            # print("Transferred model to GPU")

        dataset = FastaBatchedDataset(data.id, data.sequence)
        batches = dataset.get_batch_indices(toks_per_batch=1, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(),
                                                  batch_sampler=None)

        # scaler = MinMaxScaler()
        repr_layers = model.num_layers
        embeddings = []
        contact_maps = []

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader),
                                                        total=len(data_loader),
                                                        desc="Generating esm2 embeddings"):
                if torch.cuda.is_available() and not no_gpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                result = model(toks, repr_layers=[repr_layers], return_contacts=use_esm2_contact_map)
                representation = result["representations"][repr_layers]

                for i, label in enumerate(labels):
                    layer_for_i = representation[i, 1:len(strs[i]) + 1]

                    reduced_features = np.array(reduced_features)
                    if len(reduced_features) > 0:
                        layer_for_i = layer_for_i[:, reduced_features]

                    embedding = layer_for_i.cpu().numpy()
                    embeddings.append(embedding)

                    if use_esm2_contact_map:
                        contact_map = result["contacts"][0]
                        contact_map = contact_map.cpu().numpy()
                        contact_maps.append(contact_map)
        return embeddings, contact_maps

    except Exception as e:
        print(f"Error in get_embeddings function: {e}")

def esm_embeddings(esm2, esm2_alphabet, peptide_sequence_list, batch_size=4):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm2 = esm2.eval().to(device)
    batch_converter = esm2_alphabet.get_batch_converter()
    
    # Initialize results storage
    embeddings_results = collections.defaultdict(list)
    sequence_representations = []
    
    # Process in batches
    for i in tqdm(range(0, len(peptide_sequence_list), batch_size), desc="Processing ESM embeddings"):
        batch_sequences = peptide_sequence_list[i:i+batch_size]
        
        try:
            # Convert and move to device
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != esm2_alphabet.padding_idx).sum(1)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():  # Mixed precision for memory savings
                    results = esm2(batch_tokens, repr_layers=[33], return_contacts=False)
            
            # Move representations to CPU immediately
            token_representations = results["representations"][33].cpu()
            
            # Process each sequence in the batch
            for j, tokens_len in enumerate(batch_lens):
                seq_rep = token_representations[j, 1:tokens_len-1].mean(0)
                sequence_representations.append(seq_rep)
            
            # Clean up
            del batch_labels, batch_strs, batch_tokens, results, token_representations
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Memory error with batch size {batch_size}. Reducing batch size...")
                return esm_embeddings(esm2, esm2_alphabet, peptide_sequence_list, batch_size=max(1, batch_size//2))
            raise
    
    # Convert to DataFrame
    for i, seq_rep in enumerate(sequence_representations):
        embeddings_results[i] = seq_rep.tolist()
    
    return pd.DataFrame(embeddings_results).T

def generate_esm_embeddings(model_esm, alphabet_esm, sequence_list, output_file_path): # Renamed args
    """
    Generate ESM embeddings for a list of sequences and save the results to a CSV file.
    Input sequence_list: list of sequence strings.
    Output: Pandas DataFrame of embeddings, indexed by an ID derived from sequence or index.
    """
    # ESM model expects list of tuples: [(name1, seq1), (name2, seq2), ...]
    # Create unique names/IDs for each sequence if not provided.
    # Using "seq_INDEX" as a simple ID.
    peptide_tuples_for_esm = []
    for i, seq_str in enumerate(sequence_list):
        peptide_tuples_for_esm.append((f"seq_{i}", seq_str))

    # Process in batches if sequence_list is very large to manage memory
    # For now, processing all at once as in original code.
    # The esm_embeddings function itself might handle batching internally via batch_converter,
    # but the input to esm_embeddings is one list of tuples.
    
    # Call the internal esm_embeddings function that does the conversion
    # This function returns a DataFrame.
    embeddings_df = esm_embeddings(model_esm, alphabet_esm, peptide_tuples_for_esm)
    
    # Save to CSV
    # Ensure directory for output_file_path exists
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_csv(output_file_path) # DataFrame saves with its index
    print(f"Saved ESM embeddings to {output_file_path}")

    return embeddings_df # Return the DataFrame