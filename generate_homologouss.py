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
from vocabulary_builders import WordVocab
from utilss import *
from molecular_interactions import MolecularInteractionDataset
from sklearn.model_selection import KFold
from interaction_networks import *
from graphmakers import *
from torch import nn as nn
from rdkit.Chem import EnumerateStereoisomers

# Load ESM3 model to generate protein homologous sequences
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import torch

# Set environment variable to load local model
os.environ['INFRA_PROVIDER'] = 'EasyFormer'

# Load model
model_esm = ESM3.from_pretrained("esm3-sm-open-v1").cuda()

def generate_stereoisomers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol))
    isomer_smiles = [Chem.MolToSmiles(isomer) for isomer in isomers]
    # Ensure generate 10 isomers, repeat if not enough
    if len(isomer_smiles) < 10:
        isomer_smiles = isomer_smiles * (10 // len(isomer_smiles) + 1)
    return isomer_smiles[:10]


def dynamic_time_warping(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dtw_matrix = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dtw_matrix[i, j] = 0
            else:
                cost = 0 if seq1[i-1] == seq2[j-1] else 1
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix


def generate_sequence(masked_sequence, original_sequence):
    protein = ESMProtein(sequence=masked_sequence)
    protein = model_esm.generate(protein, GenerationConfig(track="sequence", num_steps=15, temperature=0.8))
    generated_sequence = protein.sequence
    
    # Ensure the generated sequence has the same length as the original
    if len(generated_sequence) != len(original_sequence):
        dtw_matrix = dynamic_time_warping(generated_sequence, original_sequence)
        i, j = len(generated_sequence), len(original_sequence)
        aligned_generated = []
        aligned_original = []
        while i > 0 and j > 0:
            if dtw_matrix[i, j] == dtw_matrix[i-1, j-1] + (0 if generated_sequence[i-1] == original_sequence[j-1] else 1):
                aligned_generated.append(generated_sequence[i-1])
                aligned_original.append(original_sequence[j-1])
                i -= 1
                j -= 1
            elif dtw_matrix[i, j] == dtw_matrix[i-1, j] + 1:
                aligned_generated.append(generated_sequence[i-1])
                aligned_original.append('_')
                i -= 1
            else:
                aligned_generated.append('_')
                aligned_original.append(original_sequence[j-1])
                j -= 1
        
        # Ensure the generated sequence has the same length as the original
        while len(aligned_generated) < len(original_sequence):
            aligned_generated.append('_')
        while len(aligned_original) < len(original_sequence):
            aligned_original.append('_')
        
        # Take the aligned part of the generated sequence
        generated_sequence = ''.join(aligned_generated[:len(original_sequence)])
    
    return generated_sequence

def generate_homologous_sequences(sequence):
    homologous_sequences = []
    original_length = len(sequence)
    for mask_index in range(10):
        masked_sequence = mask_sequence(sequence, mask_index, original_length)
        generated_sequence = generate_sequence(masked_sequence, sequence)
        homologous_sequences.append(generated_sequence)
    return homologous_sequences

def mask_sequence(sequence, mask_index, original_length):
    length = len(sequence)
    mask_length = original_length // 10
    start = mask_index * mask_length
    end = start + mask_length
    if end > length:
        end = length
    masked_sequence = sequence[:start] + '_' * mask_length + sequence[end:]
    return masked_sequence