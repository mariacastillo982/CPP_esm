import logging
from typing import List
import numpy as np
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from graph.tertiary_structure_handler import predict_tertiary_structures, load_tertiary_structures
from graph.edge_construction_functions import EdgeConstructionContext
from utils.scrambling import random_coordinate_matrix


def get_edges(tertiary_structure_method, sequences, pdb_path):
    if tertiary_structure_method:
        atom_coordinates_matrices = predict_tertiary_structures(sequences, pdb_path)
    else:
        atom_coordinates_matrices = load_tertiary_structures(sequences, pdb_path)

    adjacency_matrices, weights_matrices = _construct_edges(atom_coordinates_matrices,
                                                            sequences)

    return adjacency_matrices, weights_matrices

def _construct_edges(atom_coordinates_matrices, sequences):
    edge_construction_functions="distance_based_threshold"
    distance_function="euclidean"
    distance_threshold=10
    use_edge_attr=True
    num_cores = multiprocessing.cpu_count()

    esm2_contact_maps = [None] * len(atom_coordinates_matrices)

    args = [(edge_construction_functions,
             distance_function,
             distance_threshold,
             atom_coordinates,
             sequence,
             esm2_contact_map,
             use_edge_attr
             ) for (atom_coordinates, sequence, esm2_contact_map) in
            zip(atom_coordinates_matrices, sequences, esm2_contact_maps)]

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(args)), total=len(args), desc="Generating adjacency matrices", disable=False) as progress:
            futures = []
            for arg in args:
                future = pool.submit(EdgeConstructionContext.compute_edges, arg)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            adjacency_matrices = [future.result()[0] for future in futures]
            weights_matrices = [future.result()[1] for future in futures]

    return adjacency_matrices, weights_matrices
