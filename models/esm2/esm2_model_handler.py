import numpy as np
import pandas as pd
import torch
from torch import hub
import esm
from esm import FastaBatchedDataset
import os
from tqdm import tqdm
from utils import json_parser


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

def esm_embeddings(esm2, esm2_alphabet, peptide_sequence_list_tuples): # Input changed to list of tuples
  # peptide_sequence_list_tuples should be like [('prot1', 'SEQ1'), ('prot2', 'SEQ2')]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    esm2 = esm2.eval().to(device)

    batch_converter = esm2_alphabet.get_batch_converter()

    # load the peptide sequence list into the bach_converter
    # peptide_sequence_list_tuples is already in the format [ (name, seq), ... ]
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list_tuples)
    batch_lens = (batch_tokens != esm2_alphabet.padding_idx).sum(1) # Use esm2_alphabet here

    batch_tokens = batch_tokens.to(device)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
      # Here we export the last layer of the EMS model output as the representation of the peptides
      # model'esm2_t33_650M_UR50D' has 33 layers.
        results = esm2(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
    # Store embeddings in a DataFrame, indexed by the original labels/names
    # embeddings_results = collections.defaultdict(list) # Not needed if constructing DataFrame directly
    
    # Check if batch_labels (protein names) were correctly passed and use them for DataFrame index
    # If peptide_sequence_list_tuples was [('ID1', 'SEQ1'), ('ID2', 'SEQ2'), ...],
    # then batch_labels should be ['ID1', 'ID2', ...]
    
    # df_data = {label: rep.tolist() for label, rep in zip(batch_labels, sequence_representations)}
    # embeddings_df = pd.DataFrame.from_dict(df_data, orient='index')

    # If batch_labels are not protein names but just indices, then create a simple list of lists/arrays
    embedding_data_list = [seq_rep.tolist() for seq_rep in sequence_representations]
    
    # If batch_labels are indeed the names/IDs from the input tuples:
    if len(batch_labels) == len(embedding_data_list):
        embeddings_df = pd.DataFrame(embedding_data_list, index=batch_labels)
    else: # Fallback if batch_labels are not as expected
        embeddings_df = pd.DataFrame(embedding_data_list)


    del batch_strs, batch_tokens, results, token_representations, batch_lens
    gc.collect() # Explicit garbage collection
    return embeddings_df

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