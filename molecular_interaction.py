# molecular_interactions.py
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os
from sklearn.neighbors import NearestNeighbors

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class MolecularInteractionDataset(InMemoryDataset):
    def __init__(self, root, path, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len, target_seq):
        super(MolecularInteractionDataset, self).__init__(root)
        self.path = path
        df = pd.read_csv(self.path) 
        self.data_list = []
        self.process(df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len, target_seq)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['processed_data.pt']

    def download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len, target_seq):
        data_list = []
        for i in tqdm(range(len(df))):
            sm = df.loc[i, 'compound_iso_smiles']
            target_key = df.loc[i, 'target_key']
            label = df.loc[i, 'affinity']
            sm_g = smiles_graph[sm]
            ta_g = target_graph[target_key]

            
            if len(ta_g.shape) == 1:
                ta_g = ta_g.reshape(-1, 2)
            if len(sm_g.shape) == 1:
                sm_g = sm_g.reshape(-1, 2)

            # Step 1: Original graph
            com_adj_step1 = np.concatenate((ta_g, sm_g), axis=0)
            total_len_step1 = target_len[target_key] + len(smiles_idx[sm])
            tem1 = np.zeros([total_len_step1, 2])
            tem2 = np.zeros([total_len_step1, 2])
            for i in range(total_len_step1):
                tem1[i, 0] = total_len_step1
                tem1[i, 1] = i
                tem2[i, 1] = total_len_step1
                tem2[i, 0] = i
            tem1 = np.int64(tem1)
            tem2 = np.int64(tem2)
            com_adj_step1 = np.concatenate((com_adj_step1, tem1), axis=0)
            com_adj_step1 = np.concatenate((com_adj_step1, tem2), axis=0)
            com_adj_step1 = np.concatenate((com_adj_step1, [[total_len_step1, total_len_step1]]), axis=0)

            # Step 3: New nodes for step 3 using adjacency matrix and k-NN
            com_adj_step3, x_step3 = self.combined_adj(sm_g, target_len[target_key], step=3, k=2)
            com_adj_step3 = np.concatenate((ta_g, com_adj_step3), axis=0)
            total_len_step3 = target_len[target_key] + len(com_adj_step3)
            tem1 = np.zeros([total_len_step3, 2])
            tem2 = np.zeros([total_len_step3, 2])
            for i in range(total_len_step3):
                tem1[i, 0] = total_len_step3
                tem1[i, 1] = i
                tem2[i, 1] = total_len_step3
                tem2[i, 0] = i
            tem1 = np.int64(tem1)
            tem2 = np.int64(tem2)
            com_adj_step3 = np.concatenate((com_adj_step3, tem1), axis=0)
            com_adj_step3 = np.concatenate((com_adj_step3, tem2), axis=0)
            com_adj_step3 = np.concatenate((com_adj_step3, [[total_len_step3, total_len_step3]]), axis=0)

            # Step 5: New nodes for step 5 using adjacency matrix and k-NN
            com_adj_step5, x_step5 = self.combined_adj(sm_g, target_len[target_key], step=5, k=2)
            com_adj_step5 = np.concatenate((ta_g, com_adj_step5), axis=0)
            total_len_step5 = target_len[target_key] + len(com_adj_step5)
            tem1 = np.zeros([total_len_step5, 2])
            tem2 = np.zeros([total_len_step5, 2])
            for i in range(total_len_step5):
                tem1[i, 0] = total_len_step5
                tem1[i, 1] = i
                tem2[i, 1] = total_len_step5
                tem2[i, 0] = i
            tem1 = np.int64(tem1)
            tem2 = np.int64(tem2)
            com_adj_step5 = np.concatenate((com_adj_step5, tem1), axis=0)
            com_adj_step5 = np.concatenate((com_adj_step5, tem2), axis=0)
            com_adj_step5 = np.concatenate((com_adj_step5, [[total_len_step5, total_len_step5]]), axis=0)

            smiles = smiles_emb[sm]
            protein = target_emb[target_key]  
            smiles_lengths = smiles_len[sm]
            protein_lengths = target_len[target_key]

            
            if not isinstance(smiles, torch.Tensor):
                smiles = torch.tensor(smiles, dtype=torch.long)
            if not isinstance(protein, torch.Tensor):
                protein = torch.tensor(protein, dtype=torch.float32)  

            data = DATA(
                y=torch.FloatTensor([label]),
                edge_index=torch.LongTensor(com_adj_step1).transpose(1, 0),
                x=smiles.clone().detach(),
                sm=sm,
                target=target_key,
                smiles=smiles.clone().detach(),
                protein=protein.clone().detach(),
                smiles_lengths=smiles_lengths,
                protein_lengths=protein_lengths,
                seq=target_seq[target_key],
                edge_index_step3=torch.LongTensor(com_adj_step3).transpose(1, 0),
                x_step3=torch.tensor(x_step3, dtype=torch.long),
                edge_index_step5=torch.LongTensor(com_adj_step5).transpose(1, 0),
                x_step5=torch.tensor(x_step5, dtype=torch.long)
            )
            data_list.append(data)

        self.data_list = data_list
        self.save_data(data_list, self.processed_paths[0])

    def combined_adj(self, adj, size, step=3, k=2):
        adj1 = adj.copy()
        new_nodes = []
        new_adj = []
        x_step = []

        # Generate new nodes by merging consecutive nodes
        for i in range(0, len(adj1), step):
            new_node = f"new_node_{i//step}"
            new_nodes.append(new_node)
            x_step.append(i//step)
            new_adj.append([i//step, i//step])

        # Generate edges between new nodes based on adjacency matrix rules
        if len(new_nodes) > 1:
            features = np.array(x_step).reshape(-1, 1)
            n_neighbors = min(k + 1, len(features))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(features)
            distances, indices = nbrs.kneighbors(features)

            for i in range(len(new_nodes)):
                for j in indices[i][1:]:  # Exclude the node itself
                    if j < len(new_nodes):  # Ensure index is valid
                        start_i = i * step
                        end_i = start_i + step
                        start_j = j * step
                        end_j = start_j + step

                        if end_i > len(adj1) or end_j > len(adj1):
                            continue

                        sub_matrix = adj1[start_i:end_i, start_j:end_j]
                        count = np.sum(sub_matrix)

                        if (step == 3 and count > 4) or (step == 5 and count > 12):
                            new_adj.append([i, j])
                            new_adj.append([j, i])  # Add reverse edge for undirected graph

        return np.array(new_adj), x_step

    def save_data(self, data_list, path):
        torch.save(data_list, path)

    def load_data(self, path):
        return torch.load(path)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]