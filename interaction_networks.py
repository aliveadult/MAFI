import numpy as np
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from vocabulary_builders import WordVocab
import pandas as pd
import torch 
import os
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data

# Set environment variable to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set CUDA visible device to GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialFeatureAggregator(nn.Module):
    def __init__(self, groups=32):
        super(SpatialFeatureAggregator, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1))
        self.bias = Parameter(torch.ones(1, groups, 1))
        self.sig = nn.Sigmoid()
    
    def forward(self, x):  # (b, c, h)
        b, c, h = x.size()
        x = x.view(b * self.groups, -1, h)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h)
        x = x * self.sig(t)
        x = x.view(b, c, h)
        return x

class HeterogeneousAttentionLayer(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(HeterogeneousAttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2)  # (B, heads, seq_len)
        value = x  # (B, seq_len, hidden_dim)

        minus_inf = -9e15 * torch.ones_like(query)  # (B, heads, seq_len)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B, heads, seq_len)
        a = self.softmax(e)  # (B, heads, seq_len)

        out = torch.matmul(a, value)  # (B, heads, seq_len) * (B, seq_len, hidden_dim) = (B, heads, hidden_dim)
        out = torch.mean(out, dim=1).squeeze()  # (B, hidden_dim)
        return out, a
    
class MolecularGraphNetwork(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=128, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(MolecularGraphNetwork, self).__init__()

        dim = 256
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        nn6 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv6 = GINConv(nn6)
        self.bn6 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        x = F.relu(self.conv1(x, edge_index))  # (B, seq_len, hidden_dim)
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))  # (B, seq_len, hidden_dim)
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))  # (B, seq_len, hidden_dim)
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))  # (B, seq_len, hidden_dim)
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))  # (B, seq_len, hidden_dim)
        x = self.bn5(x)
        x = F.relu(self.conv6(x, edge_index))
        x = self.bn6(x)
        x = global_add_pool(x, batch)  # (B, hidden_dim)
        x = F.relu(self.fc1_xd(x))  # (B, hidden_dim)
        x = F.dropout(x, p=0.2, training=self.training)  # (B, hidden_dim)
        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


class CrossModalInteractionNet(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5, target_seq=None, smiles_emb=None, target_emb=None, smiles_len=None, target_len=None, smiles_idx=None, smiles_graph=None, target_graph=None):
        super(CrossModalInteractionNet, self).__init__()
        self.target_seq = target_seq
        self.smiles_emb = smiles_emb
        self.target_emb = target_emb
        self.smiles_len = smiles_len
        self.target_len = target_len
        self.smiles_idx = smiles_idx
        self.smiles_graph = smiles_graph
        self.target_graph = target_graph
        self.is_bidirectional = True
        # drugs
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)

        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance1 = SpatialFeatureAggregator(groups=20)
        self.out_attentions3 = HeterogeneousAttentionLayer(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.is_bidirectional = True
        self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)
   
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance2 = SpatialFeatureAggregator(groups=200)
        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = HeterogeneousAttentionLayer(hidden_dim, n_heads)

        # link
        self.out_attentions = HeterogeneousAttentionLayer(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)

        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

        # Point-wise Feed Forward Network
        self.pwff_1 = nn.Linear(hidden_dim * 3, hidden_dim * 4)
        self.pwff_2 = nn.Linear(hidden_dim * 4, hidden_dim * 3)
   
    def forward(self, data, reset=False):
        batchsize = len(data.sm)
        seq_len = 540
        tar_len = 1000
        smiles = torch.zeros(batchsize, seq_len).to(device).long()
        protein = torch.zeros(batchsize, tar_len).to(device).long()
        smiles_lengths = []
        protein_lengths = []

        for i in range(batchsize):
            sm = data.sm[i]
            seq_id = data.target[i]
            seq = self.target_seq[seq_id]
            smiles[i] = self.smiles_emb[sm]
            protein[i] = self.target_emb[seq]
            smiles_lengths.append(self.smiles_len[sm])
            protein_lengths.append(self.target_len[seq])
            sm_g = self.smiles_graph[sm]  # Use the smiles_graph in the model
            ta_g = self.target_graph[seq_id]  # Use the target_graph in the model
            sm_idx = self.smiles_idx[sm]  # Use the smiles_idx in the model

        smiles = self.smiles_embed(smiles)  # B * seq len * emb_dim
        smiles = self.smiles_input_fc(smiles)  # B * seq len * lstm_dim
        smiles = self.enhance1(smiles)   

        protein = self.protein_embed(protein)  # B * tar_len * emb_dim
        protein = self.protein_input_fc(protein)  # B * tar_len * lstm_dim
        protein = self.enhance2(protein)  
   
        smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        smiles = self.ln1(smiles)
        protein, _ = self.protein_lstm(protein)  # B * tar_len * lstm_dim *2
        protein = self.ln2(protein)

        if reset:
            return smiles, protein

        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)  # B * head* seq len
        
        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)  # B * head * tar_len


        smiles_out, smile_attn = self.out_attentions3(smiles, smiles_mask)  # B * lstm_dim*2
        protein_out, prot_attn = self.out_attentions2(protein, protein_mask)  # B * (lstm_dim *2)

        # drugs and proteins
        out_cat = torch.cat((smiles, protein), dim=1)  # B * head * lstm_dim *2
        out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # B * tar_len+seq_len * (lstm_dim *2)
        out_cat, out_attn = self.out_attentions(out_cat, out_masks)
        out = torch.cat([smiles_out, protein_out, out_cat], dim=-1)  # B * (rnn*2 *3)

        # Point-wise Feed Forward Network
        pwff = self.pwff_1(out)
        pwff = nn.ReLU()(pwff)
        pwff = self.dropout(pwff)  
        pwff = self.pwff_2(pwff)
        
        out = pwff + out 

        out = self.dropout(self.relu(self.out_fc1(out)))  # B * (256*8)
        out = self.dropout(self.relu(self.out_fc2(out)))  # B *  hidden_dim*2

        out = self.out_fc3(out).squeeze()

        del smiles_out, protein_out

        return out

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)