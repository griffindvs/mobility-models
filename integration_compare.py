#!/usr/bin/env python
# coding: utf-8

# Griffin Davis, The University of Texas at Dallas
# (C) 2022
# Data source:
# Chetty, Raj; Friedman, John; Hendren, Nathaniel; Jones, Maggie R.; Porter, Sonya R., 2022, 
# "Replication Data for: The Opportunity Atlas: Mapping the Childhood Roots of Social Mobility", 
# https://doi.org/10.7910/DVN/NKCQM1, Harvard Dataverse, V1, UNF:6:wwWmCZy1LUqtq02qHdCKFQ== [fileUNF] 

import os
import time
import sys
import pandas as pd
import numpy as np
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split

from scipy import integrate

import torch
from torch import nn
from torch.distributions import Normal
nnF = nn.functional

# Download data
get_ipython().system('wget -nc https://personal.utdallas.edu/~gcd/data/tract_merged.csv')
ds = pd.read_csv('tract_merged.csv')

# Get subset of columns
cols = ['id', 'hhinc_mean2000', 'mean_commutetime2000', 'frac_coll_plus2000', 'frac_coll_plus2010', 
        'med_hhinc1990', 'med_hhinc2016', 'popdensity2000', 'poor_share2010', 'poor_share2000', 
        'poor_share1990', 'gsmn_math_g3_2013', 'traveltime15_2010', 'emp2000', 'singleparent_share1990',
        'singleparent_share2010', 'singleparent_share2000', 'mail_return_rate2010', 'jobs_total_5mi_2015', 
        'jobs_highpay_5mi_2015', 'popdensity2010', 'ann_avg_job_growth_2004_2013', 'job_density_2013',
        'kfr_pooled_pooled_p1', 'kfr_pooled_pooled_p25', 'kfr_pooled_pooled_p50', 'kfr_pooled_pooled_p75', 'kfr_pooled_pooled_p100']

# Greater than 15k rows with missing data
excluded = ['rent_twobed2015', 'ln_wage_growth_hs_grad']

# Handle null data
ds = ds[ds.columns[ds.columns.isin(cols)]]

# Remove columns with >= 3 missing variables
dropped = 0
total_before = ds.shape[0]
for i, row in ds.iterrows():
    if row.isna().sum() >= 3: 
        dropped += 1
        ds.drop(index=i, inplace=True)
ds.reset_index()

print(f"Dropped {dropped} rows, {round((dropped/total_before)*100, 2)}% of available data")

# Shuffle split the data into training and test sets (75% / 25%)
train, test = train_test_split(ds)

train_X = train.loc[:,'hhinc_mean2000':'job_density_2013']
train_X = (train_X - train_X.mean()) / train_X.std()

percentiles = ['kfr_pooled_pooled_p1', 'kfr_pooled_pooled_p25', 'kfr_pooled_pooled_p50', 'kfr_pooled_pooled_p75']
train_Y = train.loc[:, percentiles[0]]

# Reset indexes and convert Y to pd.Series
train_X.reset_index(drop=True, inplace=True)
train_Y = train_Y.reset_index(drop=True).squeeze()

features = train_X.shape[1]
data_points = train_X.shape[0]

# Get missing indexes
train_Xmissing = np.isnan(train_X)
train_imissing = []
for m in range(data_points):
    row = []
    for i in range(features):
        if train_Xmissing.iloc[m][i]:
            row.append(i)
    train_imissing.append(row)

# Mixture Density Network with expectation maximization algorithm to handle missing data

# Store feature distribution parameters
feat_params = []

LOSS_SAMPLE_SIZE = 10000
loss_generator = np.random.default_rng()

def init_feat_params(X):
    # Initialize feature parameters
    for n in range(features):
        mu = np.nanmean(X.iloc[:, n])
        sigma = np.nanstd(X.iloc[:, n])
        feat_params.append((mu, sigma))

def log_p_x(i, xi):
    mu, sigma = feat_params[i]
    dist = Normal(mu, sigma)
    log_p_xi  = dist.log_prob(torch.tensor(np.double(xi)))
    return log_p_xi

class EMNet(nn.Module):
    def __init__(self, features, hidden_dim, out_dim):
        super(EMNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        params = self.seq(x)
        mu, sigma = torch.tensor_split(params, params.shape[0], dim=0)
        
        return mu, (sigma+1)**2

    def log_q(self, x, y, xmis, imis, mu, sigma):
        y_dist = Normal(mu, sigma)
        num = y_dist.log_prob(torch.tensor(np.double(y)))
        for k in range(len(imis)):
            num = num + log_p_x(imis[k], xmis[k])

        return num

    def loss(self, X, Y, imis, X_sample):
        m = 0
        def int_func(*xmis_hat):
            # Replace NaNs with xmis_hat
            f_i = 0
            # Replace missing values with current guesses
            for f in range(len(X[m])):
                if np.isnan(X[m, f]):
                    X[m, f] = xmis_hat[f_i]
                    f_i+=1
            # Run X with current guesses through NN
            mu, sigma = self.forward(torch.tensor(np.double(X[m])))
            dist = Normal(mu, sigma)
            # Get p(y|x)
            log_p_y_x = dist.log_prob(torch.tensor(np.double(Y[m])))

            # Get q_m(xmis_hat)
            log_qm = self.log_q(X[m], Y[m], xmis_hat, imis[m], mu, sigma)
            return torch.exp(log_qm).item() * log_p_y_x

        for m in X_sample:
            if len(imis[m]) != 0:
                # Setup list of ranges corresponding to num of missing values
                ranges = [(-np.inf, np.inf)] * len(imis[m])
                res, err = integrate.nquad(int_func, ranges)

                # Compare with sampling
                tot_samp = torch.zeros(1, 1)
                SAMPLES = 1000
                for i in range(SAMPLES):
                    x_hat = X[m]
                    xmis = []
                    for im in imis[m]:
                        mu, sigma = feat_params[im]
                        gen_x = Normal(mu, sigma).sample().item()
                        x_hat[im] = gen_x
                        xmis.append(gen_x)

                    mu_nn, sigma_nn = self.forward(torch.tensor(np.double(x_hat)))
                    log_qm = self.log_q(x_hat, Y[m], xmis, imis[m], mu_nn, sigma_nn)

                    dist_nn = Normal(mu_nn, sigma_nn)
                    log_p_y_x = dist_nn.log_prob(torch.tensor(np.double(Y[m])))
                    tot_samp = tot_samp + torch.exp(log_qm) * (log_p_y_x)

                tot_samp = tot_samp.item() / SAMPLES

                print(f"numeric integration - sampling = {res - tot_samp}")

def run_loss(mdn, X, Y, optimizer):
    mdn.train()

    init_feat_params(X)

    X = X.to_numpy()
    Y = Y.to_numpy()
    
    # Randomly sample rows from the training data for each iteration
    X_sample = loss_generator.integers(0, high=X.shape[0], size=LOSS_SAMPLE_SIZE)
        
    loss = mdn.loss(X, Y, train_imissing, X_sample)

device = "cpu"
mdn = EMNet(features, 10, 2).double().to(device)
optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-4)

run_loss(mdn, train_X, train_Y, optimizer)

