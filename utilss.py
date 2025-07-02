import numpy as np
import subprocess
from math import sqrt
from sklearn.metrics import average_precision_score
from scipy import stats
import random
import torch
import os
from tqdm import tqdm
from rdkit import Chem
import networkx as nx


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_metrics_and_return(Y, P, dataset='kiba'):
    # aupr = get_aupr(Y, P)
    cindex = get_ci(Y, P)
    mse = get_mse(Y, P)
    rmse = get_rmse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    mae = get_mae(Y, P)
    ev = get_explained_variance(Y, P)
    rm2 = get_rm2(Y, P)
    rm = get_rm(Y, P)
    r2 = get_r2(Y, P)

    print('metrics for ', dataset)
    print('cindex:', cindex)
    print('mse:', mse)
    print('rmse:', rmse)
    print('pearson:', pearson)
    print('spearman:', spearman)
    print('mae:', mae)
    print('explained_variance:', ev)
    print('rm2:', rm2)
    print('rm:', rm)
    print('r2:', r2)

    return cindex, mse, rmse, pearson, spearman, mae, ev, rm2, rm, r2

def train(model, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.float(),  data.y.float().to(device)).float()

        loss.backward()
        optimizer.step()
def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if i != j:
                if (y[i] > y[j]):
                    z = z + 1
                    u = f[i] - f[j]
                    if u > 0:
                        S = S + 1
                    elif u == 0:
                        S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    if z != 0:
        return S / z
    else:
        return 0

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
atom_dict = {5: 'C',
             6: 'C',
             9: 'O',
             12: 'N',
             15: 'N',
             21: 'F',
             23: 'S',
             25: 'Cl',
             26: 'S',
             28: 'O',
             34: 'Br',
             36: 'P',
             37: 'I',
             39: 'Na',
             40: 'B',
             41: 'Si',
             42: 'Se',
             44: 'K',
             }

def get_aupr(Y, P, threshold=7.0):
    # print(Y.shape,P.shape)
    Y = np.where(Y >= 7.0, 1, 0)
    P = np.where(P >= 7.0, 1, 0)
    aupr = average_precision_score(Y, P)
    return aupr


def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i != j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):

    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return 1 - (mult / float(y_obs_sq * y_pred_sq))


def get_mae(y, f):
    return np.mean(np.abs(y - f))

def get_explained_variance(y, f):
    return 1 - np.var(y - f) / np.var(y)

def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P) 

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_rm(Y, P):
    y_pred_mean = np.mean(P)
    numerator = np.sum((P - y_pred_mean) ** 2)
    denominator = np.sum((Y - y_pred_mean) ** 2)
    return numerator / denominator

def get_r2(y, f):
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - f) ** 2)
    return 1 - (ss_res / ss_tot)


def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def calculate_metrics(Y, P, dataset='kiba'):
    # aupr = get_aupr(Y, P)
    cindex = get_cindex(Y, P)  # DeepDTA
    rm2 = get_rm2(Y, P)  # DeepDTA
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)
    mae = get_mae(Y, P)
    ev = get_explained_variance(Y, P)
    r2 = get_r2(Y, P)
    rm = get_rm(Y, P)
    print('metrics for ', dataset)
    # print('aupr:', aupr)
    print('cindex:', cindex)

    # print('rm2:', rm2)
    print('mse:', mse)
    print('pearson', pearson)
    print('spearman:', spearman)
    print('mae:', mae)
    print('explained_variance:', ev)
    print('r2:', r2)
    print('rm:', rm)
