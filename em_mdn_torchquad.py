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
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split

from scipy import integrate

import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader, random_split
nnF = nn.functional

from torchquad import VEGAS, set_up_backend

# Setup torchquad integration with VEGAS
# (adaptive multidimensional Monte Carlo integration)
set_up_backend("torch", data_type="float64")
vegas = VEGAS()

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

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logging.info(f"Using {device} device")

# Setup dataloader for PyTorch
class MobilityDataset(Dataset):
    """Census tract mobility dataset."""

    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe   (Pandas Dataframe) : Dataframe with both the outcome and tract covariates.
            transform   (callable, optional) : Optional transform to be applied to the sample.
        """
        self.frame = dataframe
        self.transform = transform
    
    def __len__(self):
        """Return the number of samples contained within the dataframe."""
        return len(self.frame)

    def __getitem__(self, idx):
        """Return the sample corresponding to the row specified by the given index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.frame.iloc[idx,:]
        
        outcomes = row.loc['kfr_pooled_pooled_p1':'kfr_pooled_pooled_p75'].to_numpy()
        features = row.loc['hhinc_mean2000':'job_density_2013'].to_numpy()

        sample = {'outcomes': outcomes, 'features': features}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_feature_column(self, idx):
        """Return the entire feature column specified by the index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        col = self.frame.iloc[:, idx]

        return col.to_numpy()

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        outcomes, features = sample['outcomes'], sample['features']
        return {'outcomes': torch.from_numpy(outcomes),
                'features': torch.from_numpy(features)}

# Create dataset object using PyTorch dataset
dataset = MobilityDataset(ds, ToTensor())

# Split dataset for training and testing
train_dataset, test_dataset = random_split(dataset, [0.75, 0.25], generator=torch.Generator(device='cuda'))

# Create dataloaders from split datasets
# Set batch size to desired sample size for loss calculations
LOSS_SAMPLE_SIZE = 50
train_dataloader = DataLoader(train_dataset, batch_size=LOSS_SAMPLE_SIZE, 
        shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))
test_dataloader = DataLoader(test_dataset, batch_size=LOSS_SAMPLE_SIZE, 
        shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))

features = len(train_dataset[0]['features'])
data_points = len(train_dataset)

# torchquad doesn't support inf
INF = 9999999
# just use 2 sds away from mean

# Get missing indexes
train_imissing = []
for m in range(len(train_dataset)):
    sample_feat = train_dataset[m]['features']
    feat_nan = torch.isnan(sample_feat)
    row = []
    for i in range(features):
        if feat_nan[i]:
            row.append(i)
    train_imissing.append(row)

# Mixture Density Network with expectation maximization algorithm to handle missing data

# Store feature distribution parameters
feat_params = []

def init_feat_params(dataset):
    """Initialize parameters for the normal distributions of each covariate type."""
    for n in range(features):
        mu = np.nanmean(dataset.get_feature_column(n))
        sigma = np.nanstd(dataset.get_feature_column(n))
        feat_params.append((mu, np.sqrt(sigma)))

def p_x(i, xi):
    """Returns the probability of a particular feature having a certain value."""
    mu, sigma = feat_params[i]
    dist = Normal(mu, sigma**2)
    p_xi  = torch.exp(dist.log_prob(xi))
    return p_xi

class EMNet(nn.Module):
    """Mixture density network utilizing the EM algorithm for computing model loss."""
    def __init__(self, features, hidden_dim, out_dim):
        """
        Args:
            features    (int) : The number of covariates per neighborhood used to make predictions.
            hidden_dim  (int) : The dimension of the sigmoid hidden layer of the network.
            out_dim     (int) : The desired output dimension of the network.
        """
        super(EMNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )
        # Reset storage of q() function values
        self.have_stored_q = False
        self.stored_q = torch.empty((1, 1))
    
    def forward(self, x):
        """
        Send a vector of covariates through the network to obtain a 
        predicted mean and standard deviation of the corresponding
        neighborhood's outcomes.
        """
        params = self.seq(x)
        mu, sigma = torch.tensor_split(params, params.shape[0], dim=0)
        
        return mu, (sigma+1)**2

    # Evaluate q density for EM algorithm
    # For a given data point (x, y)
    def q(self, x, y, xmis, imis, mu, sigma):
        """
        Computes q for a particular data point with a guess at the missing information.
        Args:
            x      (float tensor) : the vector of covariates
            y      (float tensor) : the reference outcome
            xmis   (float tensor) : a vector containing guesses for each missing covariate
            imis   (int tensor) : a vector containing the indexes of each missing covariate in x
            mu     (float tensor) : the mean produced by the network on the previous iteration
            sigma  (float tensor) : the standard deviation produced by the network on the previous iteration
        """
        # p(y | x) with current mu, sigma
        y_dist = Normal(mu, sigma)

        # =======================
        # Numerator of q function
        # =======================

        # p(y | x)
        num = torch.exp(y_dist.log_prob(y))

        # Product[ p(x_mis) ]
        st = time.time()
        for k in range(len(imis)):
            num *= p_x(imis[k], xmis[k])

        # =========================
        # Denominator of q function
        # =========================

        # Function for the integral
        def int_func(xmis_hat):
            results = []
            # Torchquad provides multiple rounds at a time
            for guess in range(len(xmis_hat)):
                # Get product of all p(xmis)
                prod = torch.ones(1, 1)
                for i in range(len(imis)):
                    prod *= p_x(imis[i], xmis_hat[guess][i])
                x_hat = x
                f_i = 0
                # Replace missing values with current guesses
                for f in range(len(x_hat)):
                    if torch.isnan(x_hat[f]):
                        x_hat[f] = xmis_hat[guess][f_i]
                        f_i+=1
                # Run X with current guesses through NN
                mu_hat, sigma_hat = self.forward(x_hat)
                dist = Normal(mu_hat, sigma_hat)
                # Return p(y|x)
                results.append(torch.exp(dist.log_prob(y)))
            return torch.tensor(results)

        # Don't calculate the integral every time
        if self.have_stored_q:
            return num / self.stored_q

        # Setup list of ranges corresponding to num of missing values
        ranges = torch.tensor([[-float(INF), float(INF)]] * len(imis)).to(device)
        sampling_points = 3 ** 10

        # Integrate over each xmis, -inf to inf
        # st = time.time()

        den = vegas.integrate(int_func, dim=len(imis), N=sampling_points, integration_domain=ranges)
        
        self.stored_q = den
        self.have_stored_q = True
        
        # logging.info(f"integral: {round(time.time()-st,1)}s, err={err}")

        return num / den

    # Produce gradients of feature distribution parameters
    def grad_mu_xi(self, X, Y, mu_xi, sigma_xi, feat_index, imis, X_sample):
        """
        Computes the gradient of the loss function with respect to the mean
        of a single feature distribution.
        Args:
            X          (numpy matrix) : the training input
            Y          (numpy array) : the training reference output
            mu_xi      (float) : the previous mean of this feature distribution
            sigma_xi   (float) : the previous standard deviation of this feature distribution
            feat_index (int) : the column number of this feature within the data schema
            imis       (numpy matrix) : the matrix of missing features for every data sample
            X_sample   (numpy matrix) : the current batch of training data
        """
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
        """
        Computes the gradient of the loss function with respect to the standard deviation
        of a single feature distribution.
        Args:
            X          (numpy matrix) : the training input
            Y          (numpy array) : the training reference output
            mu_xi      (float) : the previous mean of this feature distribution
            sigma_xi   (float) : the previous standard deviation of this feature distribution
            feat_index (int) : the column number of this feature within the data schema
            imis       (numpy matrix) : the matrix of missing features for every data sample
            X_sample   (numpy matrix) : the current batch of training data
        """
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

    def loss(self, X, Y, imis, outcome):
        """
        Computes the loss using the negative likelihood function of the entire model.
        Args:
            X          (float32 tensor) : the training input for the given batch
            Y          (float32 tensor) : the training reference output for the given batch
            imis       (numpy matrix) : the matrix of missing features for every data sample
        """
        m = 0
        def int_func(xmis_hat):
            missing = torch.nonzero(torch.isnan(X[m]))
            results = []
            # Torchquad provides multiple rounds at a time
            for guess in range(len(xmis_hat)):
                # Replace NaNs with xmis_hat
                f_i = 0
                # Replace missing values with current guesses
                for f in range(len(X[m])):
                    if torch.isnan(X[m][f]):
                        X[m][f] = xmis_hat[guess][f_i]
                        f_i+=1
                # Run X with current guesses through NN
                mu, sigma = self.forward(X[m])
                dist = Normal(mu, sigma)
                # Get p(y|x)
                log_p_y_x = dist.log_prob(Y[m][outcome])

                # Get q_m(xmis_hat)
                qm = self.q(X[m], Y[m][outcome], xmis_hat, missing, mu, sigma)
                print(qm)
                results.append(qm * log_p_y_x)

            return torch.tensor(results)

        sum_ints = torch.zeros(1, 1).to(device)

        for m in range(len(X)):
            missing = torch.nonzero(torch.isnan(X[m]))
            # Handle case of no missing values
            if len(missing) == 0:
                mu, sigma = self.forward(X[m])
                dist = Normal(mu, sigma)
                # Get p(y|x)
                log_p_y_x = dist.log_prob(Y[m][outcome]) # 75th percentile => [2]

                # logging.info(f"row {m} : completed, no integral")

                sum_ints += log_p_y_x
            else:
                # Setup list of ranges corresponding to num of missing values
                ranges = torch.tensor([[-float(INF),float(INF)]] * len(missing)).to(device)
                sampling_points = 3 ** 10

                start_time = time.time()
                self.have_stored_q = False

                res = vegas.integrate(int_func, dim=len(missing), N=sampling_points, integration_domain=ranges)
                logging.info(f"row {m} : completed {len(imis[m])} dim integral in {int(round(time.time()-start_time, 0))}s")
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
TRAIN_EPOCHS = 1000
FEAT_PARAM_STEP = 1e-5

# Train neural network
def train_mdn(mdn, dataloader, optimizer):
    size = len(dataloader.dataset)
    mdn.train()

    for batch, sample in enumerate(dataloader):
        start_time = time.time()
        
        X = sample['features']
        Y = sample['outcomes'] # 75th percentile => 2

        # Sent data to comptutation device
        X, Y = X.to(device), Y.to(device)

        # Calculate NN loss       
        loss = mdn.loss(X, Y, train_imissing, outcome=2)

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

        # logging.info(f"Epoch {epoch+1}/{TRAIN_EPOCHS}, loss = {round(loss.item(), 4)}, time elapsed = {int(round((time.time()-start_time)/60, 0))} min")


# Evaluate neural network
def test_mdn(mdn, dataloader):
    pass
   # mdn.eval()

    #X = X.to_numpy()
    #Y = Y.to_numpy()

    # Get missing indexes
    #Xmissing = np.isnan(X)
    #imissing = []
    #for m in range(X.shape[0]):
     #   row = []
     #   for i in range(features):
     #       if Xmissing[m, i]:
    #            row.append(i)
    #    imissing.append(row)
    
    #sq_er = []
   # with torch.no_grad():
        # Get total loss
    #    test_loss = mdn.loss(X, Y, imissing)

        # Get squared error for predictions
        # for index in range(X.shape[0]):
        #     row = X[index, :]
        #     x = torch.tensor(np.double(row))
        #     y = torch.tensor(np.double(Y[index]))

        #     mu, sigma = mdn.forward(x)

        #     sq_er.append((mu.item() - y)**2)

   # test_loss /= test_X.shape[0]
    
   # if verbose:
    #    logging.info(f"Avg test loss: {test_loss}")
        # print(f"Mean squared error: {np.mean(sq_er)}")
    
    #return test_loss

# Retrain with full training dataset (selected 10)"
mdn = EMNet(features, 10, 2).double().to(device)
optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-4)

total_time = time.time()

init_feat_params(train_dataset.dataset)
train_mdn(mdn, train_dataloader, optimizer)

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

