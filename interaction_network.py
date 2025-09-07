# interaction_networks.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpatialFeatureAggregator(nn.Module):
    def __init__(self, groups=16):
        super(SpatialFeatureAggregator, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1))
        self.bias = nn.Parameter(torch.ones(1, groups, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
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
        query = self.query(x).transpose(1, 2)
        value = x
        minus_inf = -9e15 * torch.ones_like(query)
        e = torch.where(masks > 0.5, query, minus_inf)
        a = self.softmax(e)
        out = torch.matmul(a, value)
        out = torch.mean(out, dim=1)
        return out, a


class MolecularGraphNetwork(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=128, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(MolecularGraphNetwork, self).__init__()
        dim = 256
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        nn1 = nn.Sequential(nn.Linear(num_features_xd, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)
        nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(dim)
        nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = nn.BatchNorm1d(dim)
        nn4 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = nn.BatchNorm1d(dim)
        nn5 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = nn.BatchNorm1d(dim)
        nn6 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.conv6 = GINConv(nn6)
        self.bn6 = nn.BatchNorm1d(dim)
        self.fc1_xd = nn.Linear(dim, output_dim)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        x = F.relu(self.conv1(x, edge_index)); x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index)); x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index)); x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index)); x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index)); x = self.bn5(x)
        x = F.relu(self.conv6(x, edge_index)); x = self.bn6(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc1(x)); x = F.dropout(x, p=0.2)
        x = F.relu(self.fc2(x)); x = F.dropout(x, p=0.2)
        return self.out(x).squeeze()


class CrossModalInteractionNet(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5, target_seq=None, smiles_emb=None,
                 target_emb=None, smiles_len=None, target_len=None,
                 smiles_idx=None, smiles_graph=None, target_graph=None):
        super(CrossModalInteractionNet, self).__init__()
        self.target_seq = target_seq
        self.smiles_emb = smiles_emb
        self.target_emb = target_emb
        self.smiles_len = smiles_len
        self.target_len = target_len
        self.smiles_idx = smiles_idx
        self.smiles_graph = smiles_graph
        self.target_graph = target_graph
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.n_heads = n_heads
        self.bilstm_layers = bilstm_layers
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout_rate)
        self.ln1 = nn.LayerNorm(lstm_dim * 2)
        self.enhance1 = SpatialFeatureAggregator(groups=16)
        self.out_attentions3 = HeterogeneousAttentionLayer(hidden_dim, n_heads)

        self.esm_adapter = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, lstm_dim)
        )
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers,
                                    batch_first=True, bidirectional=True,
                                    dropout=dropout_rate)
        self.ln2 = nn.LayerNorm(lstm_dim * 2)
        self.enhance2 = SpatialFeatureAggregator(groups=16)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = HeterogeneousAttentionLayer(hidden_dim, n_heads)
        self.out_attentions = HeterogeneousAttentionLayer(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)
        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.pwff_1 = nn.Linear(hidden_dim * 3, hidden_dim * 4)
        self.pwff_2 = nn.Linear(hidden_dim * 4, hidden_dim * 3)

    def forward(self, data, reset=False):
        batchsize = len(data.sm)
        seq_len = 540
        tar_len = 1000
        smiles = torch.zeros(batchsize, seq_len).to(device).long()
        protein = torch.zeros(batchsize, tar_len, 1280).to(device).float()
        smiles_lengths = []
        protein_lengths = []

        for i in range(batchsize):
            sm = data.sm[i]
            seq_id = data.target[i]
            smiles[i] = self.smiles_emb[sm][:seq_len]
            vec = self.target_emb[seq_id].to(device)
            pad_len = tar_len - vec.shape[0]
            if pad_len > 0:
                vec = torch.cat([vec, torch.zeros(pad_len, 1280).to(device)], dim=0)
            protein[i] = vec[:tar_len]
            smiles_lengths.append(self.smiles_len[sm])
            protein_lengths.append(self.target_len[seq_id])

        # smiles 
        smiles = self.smiles_embed(smiles)
        smiles = self.smiles_input_fc(smiles)
        smiles = smiles.transpose(1, 2).contiguous()
        smiles = self.enhance1(smiles)
        smiles = smiles.transpose(1, 2).contiguous()
        smiles, _ = self.smiles_lstm(smiles)
        smiles = self.ln1(smiles)

        # protein 
        protein = self.esm_adapter(protein)  # ✅ 使用esm_adapter
        protein = protein.transpose(1, 2).contiguous()
        protein = self.enhance2(protein)
        protein = protein.transpose(1, 2).contiguous()
        protein, _ = self.protein_lstm(protein)
        protein = self.ln2(protein)

        if reset:
            return smiles, protein

        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)
        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)

        smiles_out, _ = self.out_attentions3(smiles, smiles_mask)
        protein_out, _ = self.out_attentions2(protein, protein_mask)

        out_cat = torch.cat((smiles, protein), dim=1)
        out_masks = torch.cat((smiles_mask, protein_mask), dim=2)
        out_cat, _ = self.out_attentions(out_cat, out_masks)
        out = torch.cat([smiles_out, protein_out, out_cat], dim=-1)

        pwff = self.pwff_1(out)
        pwff = self.relu(pwff)
        pwff = self.dropout(pwff)
        pwff = self.pwff_2(pwff)
        out = pwff + out

        out = self.dropout(self.relu(self.out_fc1(out)))
        out = self.dropout(self.relu(self.out_fc2(out)))
        out = self.out_fc3(out).squeeze()
        return out

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1]).to(adj.device)
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len:max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)