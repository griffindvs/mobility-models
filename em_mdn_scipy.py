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
from datetime import datetime
import logging
from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split

from scipy import integrate

import torch
from torch import nn
from torch.distributions import Normal
nnF = nn.functional

if not os.path.exists('logs'):
    os.makedirs('logs')

reload(logging) # Notebook workaround
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f'logs/{datetime.now().strftime("%Y-%m-%d")}_em_mdn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

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

logging.info(f"Dropped {dropped} rows, {round((dropped/total_before)*100, 2)}% of available data")

# Model for full dataset (including some missing data)

# Shuffle split the data into training and test sets (75% / 25%)
train, test = train_test_split(ds)

train_X = train.loc[:,'hhinc_mean2000':'job_density_2013']
test_X = test.loc[:,'hhinc_mean2000':'job_density_2013']

percentiles = ['kfr_pooled_pooled_p1', 'kfr_pooled_pooled_p25', 'kfr_pooled_pooled_p50', 'kfr_pooled_pooled_p75']
train_Y = train.loc[:, percentiles[0]]
test_Y = test.loc[:, percentiles[0]]

# Reset indexes and convert Y to pd.Series
train_X.reset_index(drop=True, inplace=True)
train_Y = train_Y.reset_index(drop=True).squeeze()
test_X.reset_index(drop=True, inplace=True)
test_Y = test_Y.reset_index(drop=True).squeeze()

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

LOSS_SAMPLE_SIZE = 50
loss_generator = np.random.default_rng()

def init_feat_params(X):
    # Initialize feature parameters
    for n in range(features):
        mu = np.nanmean(X.iloc[:, n])
        sigma = np.nanstd(X.iloc[:, n])
        feat_params.append((mu, np.sqrt(sigma)))

def p_x(i, xi):
    mu, sigma = feat_params[i]
    dist = Normal(mu, sigma**2)
    p_xi  = torch.exp(dist.log_prob(torch.tensor(np.double(xi))))
    return p_xi

class EMNet(nn.Module):
    def __init__(self, features, hidden_dim, out_dim):
        super(EMNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
        self.have_stored_q = False
        self.stored_q = torch.empty((1, 1), dtype=torch.float32)
    
    def forward(self, x):
        params = self.seq(x)
        mu, sigma = torch.tensor_split(params, params.shape[0], dim=0)
        
        return mu, (sigma+1)**2

    # Evaluate q density for EM algorithm
    # For a given data point (x, y)
    def q(self, x, y, xmis, imis, mu, sigma):
        # p(y | x) with current mu, sigma
        y_dist = Normal(mu, sigma)

        # =======================
        # Numerator of q function
        # =======================

        # p(y | x)
        num = torch.exp(y_dist.log_prob(torch.tensor(np.double(y))))

        # Product[ p(x_mis) ]
        st = time.time()
        for k in range(len(xmis)):
            num *= p_x(imis[k], xmis[k])

        # =========================
        # Denominator of q function
        # =========================

        # Function for the integral
        def int_func(*xmis_hat):
            # Get product of all p(xmis)
            prod = torch.ones(1, 1)
            for i in range(len(xmis_hat)):
                prod *= p_x(imis[i], torch.tensor(xmis_hat[i]))
            x_hat = x
            f_i = 0
            # Replace missing values with current guesses
            for f in range(len(x_hat)):
                if np.isnan(x_hat[f]):
                    x_hat[f] = xmis_hat[f_i]
                    f_i+=1
            # Run X with current guesses through NN
            mu_hat, sigma_hat = self.forward(torch.tensor(np.double(x_hat)))
            dist = Normal(mu_hat, sigma_hat)
            # Return p(y|x)
            return torch.exp(dist.log_prob(torch.tensor(np.double(y)))).item()

        # Don't calculate the integral every time
        if self.have_stored_q:
            return num / self.stored_q

        # Setup list of ranges corresponding to num of missing values
        ranges = [(-np.inf, np.inf)] * len(imis)

        # Integrate over each xmis, -inf to inf
        # st = time.time()
        sampling_points = 3**8
        dimension = len(imis)

        den, err = integrate.nquad(int_func, ranges, opts={"limit": 15})
        
        self.stored_q = den
        self.have_stored_q = True
        
        # logging.info(f"integral: {round(time.time()-st,1)}s, err={err}")

        return num / den

    # Produce gradients of feature distribution parameters
    def grad_mu_xi(self, X, Y, mu_xi, sigma_xi, feat_index, imis, X_sample):
        m = 0
        def int_func(*xmis_hat):
            # Replace missing values with current guesses
            f_i = 0
            x_hat = X[m]
            for f in range(len(x_hat)):
                if np.isnan(x_hat[f]):
                    x_hat[f] = xmis_hat[f_i]
                    f_i+=1
            
            nn_mu, nn_sigma = self.forward(torch.tensor(np.double(x_hat)))
            
            xi = x_hat[feat_index]
            qm = self.q(X[m], Y[m], xmis_hat, imis[m], nn_mu, nn_sigma)[0].item()

            return qm * ((xi / (sigma_xi**2)) - (mu_xi / (sigma_xi**2)))

        sum_ints = torch.zeros(1, 1)

        for m in X_sample:
            # logging.info(f"looking at data point {m}")
            # Check if this feature is missing on this row
            if np.isnan(X[m][feat_index]):
                # Setup list of ranges corresponding to num of missing values
                ranges = [(-np.inf, np.inf)] * len(imis[m])
                res, err = integrate.nquad(int_func, ranges, opts={"limit": 15})

            else:
                # Otherwise we dont need to integrate
                xi = X[m][feat_index]
                res = (xi / (sigma_xi**2)) - (mu_xi / (sigma_xi**2))

            sum_ints += res

        return sum_ints

    def grad_sigma_xi(self, X, Y, mu_xi, sigma_xi, feat_index, imis, X_sample):
        m = 0
        def int_func(*xmis_hat):
            # Replace missing values with current guesses
            f_i = 0
            x_hat = X[m]
            for f in range(len(x_hat)):
                if np.isnan(x_hat[f]):
                    x_hat[f] = xmis_hat[f_i]
                    f_i+=1

            nn_mu, nn_sigma = self.forward(torch.tensor(np.double(x_hat)))
            
            xi = x_hat[feat_index] 
            qm = self.q(X[m], Y[m], xmis_hat, imis[m], nn_mu, nn_sigma)[0].item()

            return qm * ((-1 / sigma_xi) - ((xi - mu_xi) / sigma_xi**3))

        sum_ints = torch.zeros(1, 1)

        for m in X_sample:
            # Check if this feature is missing on this row
            if np.isnan(X[m][feat_index]):
                # Setup list of ranges corresponding to num of missing values
                ranges = [(-np.inf, np.inf)] * len(imis[m])
                res, err = integrate.nquad(int_func, ranges, opts={"limit": 15})

            else:
                # Otherwise we dont need to integrate
                xi = X[m][feat_index]
                res = ((-1 / sigma_xi) - ((xi - mu_xi) / sigma_xi**3))

            sum_ints += res

        return sum_ints

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
            qm = self.q(X[m], Y[m], xmis_hat, imis[m], mu, sigma)[0].item()
            return qm * log_p_y_x

        sum_ints = torch.zeros(1, 1)

        for m in X_sample:
            # Handle case of no missing values
            if len(imis[m]) == 0:
                mu, sigma = self.forward(torch.tensor(np.double(X[m])))
                dist = Normal(mu, sigma)
                # Get p(y|x)
                log_p_y_x = dist.log_prob(torch.tensor(np.double(Y[m])))

                # logging.info(f"row {m} : completed, no integral")

                sum_ints += log_p_y_x
            else:
                # Setup list of ranges corresponding to num of missing values
                ranges = [(-np.inf, np.inf)] * len(imis[m])
                start_time = time.time()
                self.have_stored_q = False

                res, err = integrate.nquad(int_func, ranges, opts={"limit": 15})
                logging.info(f"row {m} : completed {len(imis[m])} dim integral in {int(round(time.time()-start_time, 0))}s, err={err}")
                sum_ints += res

        return -1 * sum_ints

    def predict(self, x, imis):
        y = 0 # Replaced by for loop that calls integral function
        def int_func(xmis_hat):
            # Replace NaNs with xmis_hat
            f_i = 0
            # Replace missing values with current guesses
            for f in range(len(x)):
                if np.isnan(x[f]):
                    x[f] = xmis_hat[f_i]
                    f_i+=1
            # Run X with current guesses through NN
            mu, sigma = self.forward(torch.tensor(np.double(x)))
            dist = Normal(mu, sigma)

            # Get log(p(y|x))
            log_p_y_x = dist.log_prob(torch.tensor(np.double(y)))

            # Return p(y|x)
            return torch.exp(log_p_y_x)

        # Store the probability of each possible y
        y_probs = []

        # Iterate over subset of possible y values: 0, 0.01, 0.02, ..., 1
        for y in np.linspace(0, 1, 101):
            # Setup list of ranges corresponding to num of missing values in input vector
            ranges = [(-np.inf, np.inf) for i in range(len(imis))]
            # Integrate over missing values
            res, err = integrate.nquad(int_func, ranges, opts={"limit": 20})
            # Add result to output array
            y_probs.append((y, res.item()))

# Training constants
TRAIN_ITERATIONS = 1000
FEAT_PARAM_STEP = 1e-5

# Train neural network
def train_mdn(mdn, X, Y, optimizer, verbose):
    mdn.train()

    init_feat_params(X)

    X = X.to_numpy()
    Y = Y.to_numpy()
    
    for index in range(TRAIN_ITERATIONS):
        start_time = time.time()
        # Calculate NN loss       

        # Randomly sample rows from the training data for each iteration
        X_sample = loss_generator.integers(0, high=X.shape[0], size=LOSS_SAMPLE_SIZE)
        
        loss = mdn.loss(X, Y, train_imissing, X_sample)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update feature parameters using gradient descent
        for f in range(features):
            cur_mu, cur_sigma = feat_params[f]
            grad_mu = mdn.grad_mu_xi(X, Y, cur_mu, cur_sigma, f, train_imissing, X_sample).item() / LOSS_SAMPLE_SIZE
            grad_sigma = mdn.grad_sigma_xi(X, Y, cur_mu, cur_sigma, f, train_imissing, X_sample).item() / LOSS_SAMPLE_SIZE

            feat_params[f] = (cur_mu + (FEAT_PARAM_STEP * grad_mu), cur_sigma + (FEAT_PARAM_STEP * grad_sigma))

        if verbose:
            logging.info(f"Epoch {index+1}/{TRAIN_ITERATIONS}, loss = {round(loss.item(), 4)}, time elapsed = {int(round((time.time()-start_time)/60, 0))} min")


# Evaluate neural network
def test_mdn(mdn, X, Y, verbose):
    mdn.eval()

    X = X.to_numpy()
    Y = Y.to_numpy()

    # Get missing indexes
    Xmissing = np.isnan(X)
    imissing = []
    for m in range(X.shape[0]):
        row = []
        for i in range(features):
            if Xmissing[m, i]:
                row.append(i)
        imissing.append(row)
    
    sq_er = []
    with torch.no_grad():
        # Get total loss
        test_loss = mdn.loss(X, Y, imissing)

        # Get squared error for predictions
        # for index in range(X.shape[0]):
        #     row = X[index, :]
        #     x = torch.tensor(np.double(row))
        #     y = torch.tensor(np.double(Y[index]))

        #     mu, sigma = mdn.forward(x)

        #     sq_er.append((mu.item() - y)**2)

    test_loss /= test_X.shape[0]
    
    if verbose:
        logging.info(f"Avg test loss: {test_loss}")
        # print(f"Mean squared error: {np.mean(sq_er)}")
    
    return test_loss


# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Retrain with full training dataset (selected 10)"
mdn = EMNet(features, 10, 2).double().to(device)
optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-4)

total_time = time.time()

train_mdn(mdn, train_X, train_Y, optimizer, True)

logging.info(f"Training completed in {(time.time()-total_time)/60:.{4}f}min")

# Save the final model
torch.save(mdn, f'em_mdn.pt')
logging.info("Neural network saved to em_mdn.pt")

# Save feature distributions to disk
feat_dist_file = open('em_mdn_feat.txt', 'w')

for d in feat_params:
    feat_dist_file.write(f"{d[0]}, {d[1]}\n")

feat_dist_file.close()

logging.info("Feature distributions saved to em_mdn_feat.txt")

