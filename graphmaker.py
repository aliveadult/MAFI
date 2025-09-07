import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import EnumerateStereoisomers
import networkx as nx

import torch
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from vocabulary_builder import WordVocab
from utils import *

from molecular_interaction import MolecularInteractionDataset
from sklearn.model_selection import KFold
from interaction_network import *
from torch import nn as nn

def build_smiles_contact_map(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    if len(edge_index.shape) == 1:
        edge_index = edge_index.reshape(-1, 2)
    return c_size, edge_index


def build_protein_contact_map(target_key, target_sequence, contact_dir, start, end):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map = contact_map[start:end, start:end]
    index_row, index_col = np.where(contact_map > 0.8)

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    if len(target_edge_index.shape) == 1:
        target_edge_index = target_edge_index.reshape(-1, 2)
    return target_size, target_edge_index