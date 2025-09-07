# main.py  —— Strict dual cold-start: no duplicated drugs or proteins between train/test
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from utils import *
from graphmaker import *
from molecular_interaction import MolecularInteractionDataset
from interaction_network import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Hyper-parameters ----------
LR = 1e-4
NUM_EPOCHS = 150
seed = 42
dataset__name = 'davis'            # <--- only change here
batch_size = 128

# ---------- Auto paths ----------
root_dir = '/media/6t/hanghuaibin/Easyformer'
dataset_csv = f'{root_dir}/{dataset__name}_processed.csv'
contact_dir = f'{root_dir}/{dataset__name}_npy_contact_maps'
esm_dir = f'{root_dir}/{dataset__name}_protein_esm2_all_embeddings'
vocab_path = f'{root_dir}/Vocab/smiles_vocab.pkl'
model_save_dir = f'{root_dir}/Model'
os.makedirs(model_save_dir, exist_ok=True)

# ---------- Load data ----------
df = pd.read_csv(dataset_csv)
smiles_set = set(df['compound_iso_smiles'])
target_keys = set(df['target_key'])
target_seq = dict(zip(df['target_key'], df['target_sequence']))

# ---------- Build graphs ----------
smiles_graph = {}
for sm in smiles_set:
    _, graph = build_smiles_contact_map(sm)
    smiles_graph[sm] = graph

target_graph = {}
for tgt in target_keys:
    start, end = 0, len(target_seq[tgt])
    _, graph = build_protein_contact_map(tgt, target_seq[tgt], contact_dir, start, end)
    target_graph[tgt] = graph

# ---------- Small-molecule tokens ----------
from vocabulary_builder import WordVocab
drug_vocab = WordVocab.load_vocab(vocab_path)
seq_len = 540
smiles_idx, smiles_emb, smiles_len = {}, {}, {}
for sm in smiles_set:
    content, flag = [], 0
    for i in range(len(sm)):
        if flag >= len(sm): break
        if flag + 1 < len(sm) and sm[flag:flag + 2] in drug_vocab.stoi:
            content.append(drug_vocab.stoi[sm[flag:flag + 2]])
            flag += 2
            continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag += 1
    if len(content) > seq_len:
        content = content[:seq_len]
    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    if len(X) > seq_len:
        X = X[:seq_len]
    else:
        X.extend([drug_vocab.pad_index] * (seq_len - len(X)))
    smiles_len[sm] = len(content)
    smiles_emb[sm] = torch.tensor(X)
    smiles_idx[sm] = [i for i, c in enumerate(X) if atom_dict.get(c) is not None]

# ---------- Protein ESM features ----------
target_emb = {}
target_len = {}
for k in target_keys:
    vec_list = [torch.load(os.path.join(esm_dir, f"{k}.pt"))]
    for i in range(1, 11):
        path = os.path.join(esm_dir, f"{k}_homolog_{i}.pt")
        vec_list.append(torch.load(path) if os.path.exists(path) else vec_list[0])
    max_len = max(v.shape[0] for v in vec_list)
    padded = [
        torch.cat([v, torch.zeros(max_len - v.shape[0], 1280)]) if v.shape[0] < max_len else v[:max_len]
        for v in vec_list
    ]
    stacked = torch.stack(padded)
    weights = torch.softmax(torch.randn(11), dim=0).view(11, 1, 1)
    fused = (stacked * weights).sum(0)
    target_emb[k] = fused
    target_len[k] = fused.shape[0]

# ---------- Build dataset ----------
dataset = MolecularInteractionDataset(
    root=root_dir,
    path=dataset_csv,
    smiles_emb=smiles_emb,
    target_emb=target_emb,
    smiles_idx=smiles_idx,
    smiles_graph=smiles_graph,
    target_graph=target_graph,
    smiles_len=smiles_len,
    target_len=target_len,
    target_seq=target_seq
)

# ---------- Strict dual cold-start split ----------
np.random.seed(seed)

# 1. Shuffle drug and protein lists
all_drugs = np.array(list(smiles_set))
all_prots = np.array(list(target_keys))
np.random.shuffle(all_drugs)
np.random.shuffle(all_prots)

# 2. 20% drugs + 20% proteins as test set
test_drug_ratio = 0.20
test_prot_ratio = 0.20
n_drug_test = int(len(all_drugs) * test_drug_ratio)
n_prot_test = int(len(all_prots) * test_prot_ratio)

test_drugs = set(all_drugs[:n_drug_test])
test_prots = set(all_prots[:n_prot_test])
train_drugs = set(all_drugs) - test_drugs
train_prots = set(all_prots) - test_prots

# 3. Assign samples based on whether drug&protein are both in train/test
def idx_by_split(df, drug_set, prot_set):
    mask = (df['compound_iso_smiles'].isin(drug_set)) & (df['target_key'].isin(prot_set))
    return df.index[mask].values

train_idx = idx_by_split(df, train_drugs, train_prots)
test_idx  = idx_by_split(df, test_drugs,  test_prots)

# 4. Split 12.5% from training samples for validation
train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=seed)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset   = torch.utils.data.Subset(dataset, val_idx)
test_dataset  = torch.utils.data.Subset(dataset, test_idx)

# ---------- Print dual cold-start info ----------
print("=== Strict Drug & Target Cold-Start Check ===")
print(f"Total drugs : {len(all_drugs)} | Total proteins : {len(all_prots)}")
print(f"Train drugs : {len(train_drugs)} | Train proteins : {len(train_prots)}")
print(f"Test  drugs : {len(test_drugs)}  | Test  proteins : {len(test_prots)}")
print(f"Train samples : {len(train_idx)} | Val samples : {len(val_idx)} | Test samples : {len(test_idx)}")
print(f"Drug  overlap : {train_drugs & test_drugs}")
print(f"Protein overlap : {train_prots & test_prots}")
print("=============================================")

# ---------- Data loaders ----------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)

# ---------- Model ----------
model = CrossModalInteractionNet(
    embedding_dim=256, lstm_dim=256, hidden_dim=512, dropout_rate=0.3,
    alpha=0.2, n_heads=16, bilstm_layers=3,
    protein_vocab=26, smile_vocab=45, theta=0.5,
    target_seq=target_seq,
    smiles_emb=smiles_emb,
    target_emb=target_emb,
    smiles_len=smiles_len,
    target_len=target_len,
    smiles_idx=smiles_idx,
    smiles_graph=smiles_graph,
    target_graph=target_graph
).to(device)

# ---------- Optimizer ----------
optimizer = torch.optim.AdamW([
    {'params': model.esm_adapter.parameters(), 'lr': 1e-4},
    {'params': model.protein_lstm.parameters(), 'lr': 5e-5},
    {'params': model.smiles_lstm.parameters(), 'lr': 5e-5},
    {'params': model.out_attentions.parameters(), 'lr': 1e-4},
    {'params': model.out_attentions2.parameters(), 'lr': 1e-4},
    {'params': model.out_attentions3.parameters(), 'lr': 1e-4},
    {'params': model.out_fc1.parameters(), 'lr': 1e-4},
    {'params': model.out_fc2.parameters(), 'lr': 1e-4},
    {'params': model.out_fc3.parameters(), 'lr': 1e-4},
    {'params': model.pwff_1.parameters(), 'lr': 1e-4},
    {'params': model.pwff_2.parameters(), 'lr': 1e-4},
], weight_decay=1e-5)

schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-4, last_epoch=-1)

# ---------- Training ----------
best_mse = 1e9
best_epoch = -1
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}")
    train(model, train_loader, optimizer, epoch)
    G, P = predicting(model, val_loader)
    val_mse = get_mse(G, P)
    if val_mse < best_mse:
        best_mse = val_mse
        best_epoch = epoch
        torch.save(model.state_dict(), f'{model_save_dir}/{dataset__name}_drug_target_cold_best.pt')
        print(f"Validation MSE improved to {val_mse:.4f} at epoch {epoch}")
    else:
        print(f"Validation MSE {val_mse:.4f}, best {best_mse:.4f}")
    schedule.step()

# ---------- Dual cold-start test ----------
model.load_state_dict(torch.load(f'{model_save_dir}/{dataset__name}_drug_target_cold_best.pt'))
G, P = predicting(model, test_loader)
cindex, mse, rmse, pearson, spearman, mae, ev, rm2, rm, r2 = calculate_metrics_and_return(G, P, dataset__name)
print("\n=== Drug & Target Cold-Start Test Results ===")
print(f"CI: {cindex:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
print(f"Pearson: {pearson:.4f}, Spearman: {spearman:.4f}, MAE: {mae:.4f}")