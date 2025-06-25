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
from generate_homologouss import *

from torch import nn as nn

# Set environment variable to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set CUDA visible device to GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 1e-4
NUM_EPOCHS = 150
seed = 0
dataset__name = 'davis'  # dataset name
batch_size = 128

# Data preprocessing part
dataset_name = '/media/6t/hanghuaibin/Easyformer/davis_processed.csv'
df = pd.read_csv(dataset_name)

smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])

target_seq = {}
for i in range(len(df)):
    target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']

smiles_graph = {}
for sm in smiles:
    _, graph = build_smiles_contact_map(sm)
    smiles_graph[sm] = graph

target_uniprot_dict = {}
target_process_start = {}
target_process_end = {}

for i in range(len(df)):
    target = df.loc[i, 'target_key']

    uniprot = df.loc[i, 'target_key']

    target_uniprot_dict[target] = uniprot
    target_process_start[target] = df.loc[i, 'target_sequence_start']
    target_process_end[target] = df.loc[i, 'target_sequence_end']

if 'kiba' in dataset__name:
    contact_dir = '/media/6t/hanghuaibin/Easyformer/kibas_npy_contact_maps'
elif 'davis' in dataset__name:
    contact_dir = '/media/6t/hanghuaibin/Easyformer/davis_npy_contact_maps'
else:
    raise ValueError("Unsupported dataset. Please check the dataset name.")

target_graph = {}

for target in tqdm(target_seq.keys()):
    uniprot = target_uniprot_dict[target]
    contact_path = os.path.join(contact_dir, uniprot + '.npy')
    if not os.path.exists(contact_path):
        raise FileNotFoundError(f"Contact map file {contact_path} not found. Please ensure all contact maps are generated.")
    contact_map = np.load(contact_path)
    start = target_process_start[target]
    end = target_process_end[target]
    _, graph = build_protein_contact_map(uniprot, target_seq[target], contact_dir, start, end)
    target_graph[target] = graph

# Load vocabulary
drug_vocab = WordVocab.load_vocab('/media/6t/hanghuaibin/Easyformer/Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('/media/6t/hanghuaibin/Easyformer/Vocab/protein_vocab.pkl')

# Build molecule features
tar_len = 1000
seq_len = 540

smiles_idx = {}
smiles_emb = {}
smiles_len = {}
for sm in smiles:
    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)):
            if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag = flag + 1

    if len(content) > seq_len:
        content = content[:seq_len]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    if not smiles_idx.__contains__(sm):
        tem = []
        for i, c in enumerate(X):
            if atom_dict.__contains__(c):
                tem.append(i)
        smiles_idx[sm] = tem

# Build protein features
target_emb = {}
target_len = {}
for k in target_seq:
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)):
            if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag = flag + 1

    if len(content) > tar_len:
        content = content[:tar_len]
    else:
        padding = [target_vocab.pad_index] * (tar_len - len(content))
        content.extend(padding)
    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)
    target_emb[seq] = torch.tensor(X)

# Build dataset
print("Building dataset...")
dataset = MolecularInteractionDataset(root='/media/6t/hanghuaibin/Easyformer/',
                                      path=dataset_name, 
                                      smiles_emb=smiles_emb, 
                                      target_emb=target_emb, 
                                      smiles_idx=smiles_idx, 
                                      smiles_graph=smiles_graph, 
                                      target_graph=target_graph, 
                                      smiles_len=smiles_len, 
                                      target_len=target_len,
                                      target_seq=target_seq)

# Model training preparation
model_name = 'default'
model_file_name = '/media/6t/hanghuaibin/Easyformer/Model/' + dataset__name + '_' + model_name + '.pt'
os.makedirs(os.path.dirname(model_file_name), exist_ok=True)

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

# Generate protein homologous sequence features
for k in target_seq:
    original_sequence = target_seq[k]
    homologous_sequences = generate_homologous_sequences(original_sequence)
    
    feature_vectors = []
    for seq in [original_sequence] + homologous_sequences:
        content = []
        flag = 0
        for i in range(len(seq)):
            if flag >= len(seq):
                break
            if (flag + 1 < len(seq)):
                if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                    content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                    flag = flag + 2
                    continue
            content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
            flag = flag + 1

        if len(content) > tar_len:
            content = content[:tar_len]
        else:
            padding = [target_vocab.pad_index] * (tar_len - len(content))
            content.extend(padding)
        X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
        if tar_len > len(X):
            padding = [target_vocab.pad_index] * (tar_len - len(X))
            X.extend(padding)
        feature_vectors.append(torch.tensor(X))

    feature_vectors = torch.stack(feature_vectors)
    weights = torch.nn.functional.softmax(torch.randn(11), dim=0)
    weighted_vector = torch.sum(feature_vectors * weights.unsqueeze(1), dim=0)
    target_emb[k] = weighted_vector

# Rebuild dataset
print("Building dataset...")
dataset = MolecularInteractionDataset(root='/media/6t/hanghuaibin/Easyformer/',
                                      path=dataset_name, 
                                      smiles_emb=smiles_emb, 
                                      target_emb=target_emb, 
                                      smiles_idx=smiles_idx, 
                                      smiles_graph=smiles_graph, 
                                      target_graph=target_graph, 
                                      smiles_len=smiles_len, 
                                      target_len=target_len,
                                      target_seq=target_seq)

# Model training and evaluation
for fold, (train_indices, test_indices) in enumerate(kf.split(dataset)):
    print(f"Fold {fold+1}")
    
    model = CrossModalInteractionNet(embedding_dim=256, 
                                     lstm_dim=256, 
                                     hidden_dim=512, 
                                     dropout_rate=0.3,
                                     alpha=0.2, 
                                     n_heads=16, 
                                     bilstm_layers=3, 
                                     protein_vocab=26,
                                     smile_vocab=45, 
                                     theta=0.5,
                                     target_seq=target_seq,    
                                     smiles_emb=smiles_emb,                                     
                                     target_emb=target_emb,
                                     smiles_len=smiles_len,
                                     target_len=target_len,
                                     smiles_idx=smiles_idx,
                                     smiles_graph=smiles_graph,
                                     target_graph=target_graph).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-4, last_epoch=-1)
    
    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_test_epoch = -1

    for epoch in range(NUM_EPOCHS):
        print("No {} epoch".format(epoch))
        
        if epoch == 0:
            val_size = int(len(dataset) * 0.1)
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset=train_dataset,
                lengths=[len(train_dataset)-val_size, val_size],
                generator=torch.Generator().manual_seed(0)
            )
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        train(model, train_loader, optimizer, epoch)
        G, P = predicting(model, val_loader)
        val1 = get_mse(G, P)
        
        if val1 < best_mse:
            best_mse = val1
            best_epoch = epoch + 1
            if model_file_name is not None:
                torch.save(model.state_dict(), model_file_name)
            print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
        else:
            print('current mse: ', val1, ' No improvement since epoch ', best_epoch, '; best_mse', best_mse)
        
        schedule.step()
    
    print(model_file_name)
    save_model = torch.load(model_file_name)
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    
    G, P = predicting(model, test_loader)
    cindex, mse, rmse, pearson, spearman, mae, ev, rm2, rm, r2 = calculate_metrics_and_return(G, P, test_loader.dataset)
    print(f"Test metrics - CI: {cindex}, MSE: {mse}, RMSE: {rmse}, Pearson: {pearson}, Spearman: {spearman}, MAE: {mae}, EV: {ev}, R2: {r2}, RM: {rm}, RM2: {rm2}")
    break