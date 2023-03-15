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

import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader, random_split
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

# Remove columns with greater than 15k rows with missing data
excluded = ['rent_twobed2015', 'ln_wage_growth_hs_grad']
ds = ds[ds.columns[ds.columns.isin(cols)]]

# Remove columns with >= 3 missing variables
dropped = 0
total_before = ds.shape[0]
for i, row in ds.iterrows():
    if row.isna().sum() >= 5: 
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

    def __init__(self, dataframe, transform=None, normalize=False):
        """
        Args:
            dataframe   (Pandas Dataframe) : Dataframe with both the outcome and tract covariates.
            transform   (callable, optional) : Optional transform to be applied to the sample.
            normalize   (bool, optional) : Normalize the columns of the data frame
        """
        self.frame = dataframe
        self.transform = transform
        
        # Normalize columns, handling NaN values, skipping id column and outcomes
        if normalize:
            self.frame.iloc[:,1:-5] = (self.frame.iloc[:,1:-5] - self.frame.iloc[:,1:-5].mean()) / self.frame.iloc[:,1:-5].std()
    
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
dataset = MobilityDataset(ds, ToTensor(), normalize=True)

# Split dataset for training and testing
train_dataset, test_dataset = random_split(dataset, [0.75, 0.25], generator=torch.Generator(device='cpu'))

# Create dataloaders from split datasets
# Set batch size to desired sample size for loss calculations
BATCH_SIZE = 50
Q_SAMPLES = 20
MIN_SD = 0.01
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=0, generator=torch.Generator(device='cpu'))
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=0, generator=torch.Generator(device='cpu'))

features = len(train_dataset[0]['features'])

# Mixture Density Network with expectation maximization algorithm to handle missing data

# Store feature distribution parameters
feat_params = []

def init_feat_params(dataset):
    """Initialize parameters for the normal distributions of each covariate type."""
    for n in range(features):
        mu = torch.tensor(np.nanmean(dataset.get_feature_column(n)), requires_grad=False).to(device)
        sigma = torch.tensor(np.nanstd(dataset.get_feature_column(n)), requires_grad=False).to(device)
        feat_params.append((mu, sigma))

def log_p_x(i, xi):
    """Returns the probability of a particular feature having a certain value."""
    mu, sigma = feat_params[i]
    dist = Normal(mu, sigma)
    log_p_xi  = dist.log_prob(xi)
    return log_p_xi

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
    def log_q(self, x, y, xmis, imis, mu, sigma):
        """
        Computes log(q) for a particular data point with a guess at the missing information.
        Args:
            x      (float tensor) : the vector of covariates
            y      (float tensor) : the reference outcome
            xmis   (float tensor) : a vector containing guesses for each missing covariate
            imis   (float tensor) : a vector containing the indexes of each missing covariate in x
            mu     (float tensor) : the mean produced by the network on the previous iteration
            sigma  (float tensor) : the standard deviation produced by the network on the previous iteration
        """
        # p(y | x) with current mu, sigma
        y_dist = Normal(mu, sigma)
        num = y_dist.log_prob(y)

        # Sum[ log p(x_mis) ]
        for k in range(len(imis)):
            num = num + log_p_x(imis[k], xmis[k])

        # We can ignore the normalizing denominator for sampling purposes
        return num

    # Produce gradients of feature distribution parameters
    # Not included in PyTorch computation graph
    @torch.no_grad()
    def grad_mu_xi(self, X, Y, mu_xi, sigma_xi, feat_index, outcome):
        """
        Computes the gradient of the loss function with respect to the mean
        of a single feature distribution.
        Args:
            X          (double tensor) : the training input
            Y          (double tensor) : the training reference output
            mu_xi      (float tensor) : the previous mean of this feature distribution
            sigma_xi   (float tensor) : the previous standard deviation of this feature distribution
            feat_index (int) : the column number of this feature within the data schema
            outcome    (int) : 0=25th percentile, 1=50th percentile, 2=75th percentile, 3=100th percentile
        """
        sum_loss = torch.zeros(1, 1).to(device)
        for m in range(len(X)):
            missing = torch.nonzero(torch.isnan(X[m]))
            # Check if this feature is missing on this row
            if torch.isnan(X[m][feat_index]):
                # Sample from feature distributions
                tot_samp = torch.zeros(1, 1).to(device)
                for i in range(Q_SAMPLES):
                    x_hat = X[m].clone()
                    xmis = []
                    for imis in missing:
                        mu, sigma = feat_params[imis.item()]
                        gen_x = Normal(mu, sigma).sample().double()
                        # Replace nan with sample
                        x_hat[imis] = gen_x
                        xmis.append(gen_x)
                    # Run sampled x through NN
                    mu_nn, sigma_nn = self.forward(x_hat)

                    xi = x_hat[feat_index]
                    # Compute q
                    qm = torch.exp(self.log_q(x_hat, Y[m][outcome], torch.tensor(xmis), missing, mu_nn, sigma_nn))
                    tot_samp = tot_samp + (qm * ((xi / (sigma_xi**2)) - (mu_xi / (sigma_xi**2))))

                # Add average tot_samp to loss
                sum_loss = sum_loss + (tot_samp / Q_SAMPLES)
            else:
                # Otherwise we can calculate directly
                xi = X[m][feat_index]
                res = (xi / (sigma_xi**2)) - (mu_xi / (sigma_xi**2))
                sum_loss = sum_loss + res

        return sum_loss

    @torch.no_grad()
    def grad_sigma_xi(self, X, Y, mu_xi, sigma_xi, feat_index, outcome):
        """
        Computes the gradient of the loss function with respect to the standard deviation
        of a single feature distribution.
        Args:
            X          (double tensor) : the training input
            Y          (double tensor) : the training reference output
            mu_xi      (float tensor) : the previous mean of this feature distribution
            sigma_xi   (float tensor) : the previous standard deviation of this feature distribution
            feat_index (int) : the column number of this feature within the data schema
            outcome    (int) : 0=25th percentile, 1=50th percentile, 2=75th percentile, 3=100th percentile
        """
        sum_loss = torch.zeros(1, 1).to(device)
        for m in range(len(X)):
            missing = torch.nonzero(torch.isnan(X[m]))
            # Check if this feature is missing on this row
            if torch.isnan(X[m][feat_index]):
                # Sample from feature distributions
                tot_samp = torch.zeros(1, 1).to(device)
                for i in range(Q_SAMPLES):
                    x_hat = X[m].clone()
                    xmis = []
                    for imis in missing:
                        mu, sigma = feat_params[imis.item()]
                        gen_x = Normal(mu, sigma).sample().double()
                        # Replace nan with sample
                        x_hat[imis] = gen_x
                        xmis.append(gen_x)
                    # Run sampled x through NN
                    mu_nn, sigma_nn = self.forward(x_hat)

                    xi = x_hat[feat_index]
                    # Compute q
                    qm = torch.exp(self.log_q(x_hat, Y[m][outcome], torch.tensor(xmis), missing, mu_nn, sigma_nn))
                    tot_samp = tot_samp + (qm * ((-1 / sigma_xi) - ((xi - mu_xi) / sigma_xi**3)))

                # Add average tot_samp to loss
                sum_loss = sum_loss + (tot_samp / Q_SAMPLES)
            else:
                # Otherwise we can calculate directly
                xi = X[m][feat_index]
                res = ((-1 / sigma_xi) - ((xi - mu_xi) / sigma_xi**3))
                sum_loss = sum_loss + res

        return sum_loss

    def loss(self, X, Y, outcome):
        """
        Computes the neural network loss using the NN component of the negative likelihood function of the entire model.
        Args:
            X          (double tensor) : the training input for the given batch
            Y          (double tensor) : the training reference output for the given batch
            outcome    (int) : 0=25th percentile, 1=50th percentile, 2=75th percentile, 3=100th percentile
        """
        sum_loss = torch.zeros(1, 1).to(device)
        for m in range(len(X)):
            missing = torch.nonzero(torch.isnan(X[m]))

            # Handle case of no missing values
            if len(missing) == 0:
                mu, sigma = self.forward(X[m])
                dist = Normal(mu, sigma)
                
                # Get p(y|x)
                log_p_y_x = dist.log_prob(Y[m][outcome])

                sum_loss = sum_loss + log_p_y_x
            else:
                # Sample from feature distributions
                tot_samp = torch.zeros(1, 1).to(device)

                for i in range(Q_SAMPLES):
                    x_hat = X[m].clone()
                    xmis = []

                    # First sample for each missing x
                    for imis in missing:
                        mu, sigma = feat_params[imis.item()]
                        gen_x = Normal(mu, sigma).sample().double()
                        # Replace nan with sample
                        x_hat[imis] = gen_x
                        xmis.append(gen_x)
                    
                    # Run sampled x through NN
                    mu_nn, sigma_nn = self.forward(x_hat)
                    
                    # Compute q
                    log_qm = self.log_q(x_hat, Y[m][outcome], torch.tensor(xmis), missing, mu_nn, sigma_nn)
                    
                    # Use q and sampled x to compute component of loss
                    dist_nn = Normal(mu_nn, sigma_nn)
                    log_p_y_x = dist_nn.log_prob(Y[m][outcome])
                    
                    # Add to total of samples
                    tot_samp = tot_samp + torch.exp(log_qm) * (log_p_y_x)

                # Add average tot_samp to loss
                sum_loss = sum_loss + (tot_samp / Q_SAMPLES)

        return -1 * sum_loss

    @torch.no_grad()
    def predict(self, x):
        # Iterate over subset of possible y values: 0, 0.01, 0.02, ..., 1
        y_probs = []
        for y in np.linspace(0, 1, 101):
            # Sample from feature distributions
            missing = torch.nonzero(torch.isnan(x))
            tot_samp = torch.zeros(1, 1).to(device)
            for i in range(Q_SAMPLES):
                x_hat = x.clone()
                xmis = []

                # First sample for each missing x
                for imis in missing:
                    mu, sigma = feat_params[imis.item()]
                    gen_x = Normal(mu, sigma).sample().double()
                    # Replace nan with sample
                    x_hat[imis] = gen_x
                    xmis.append(gen_x)
                    
                # Run sampled x through NN
                mu_nn, sigma_nn = self.forward(x_hat)
                    
                # Use sampled x to p(y | x)
                dist_nn = Normal(mu_nn, sigma_nn)
                log_p_y_x = dist_nn.log_prob(torch.tensor(y).to(device))
                    
                # Add to total of samples
                tot_samp = tot_samp + log_p_y_x

            res = torch.exp(tot_samp / Q_SAMPLES)

            # Add result to output array
            y_probs.append((y, res.item()))

        return y_probs

# Training constants
TRAIN_EPOCHS = 10
FEAT_PARAM_STEP = 1e-5

# Train neural network
def train_mdn(mdn, dataloader, optimizer, outcome):
    size = len(dataloader.dataset)
    mdn.train()

    start_time = time.time()
    for batch, sample in enumerate(dataloader):
        X = sample['features']
        Y = sample['outcomes']

        # Sent data to comptutation device
        X, Y = X.to(device), Y.to(device)

        # Calculate NN loss       
        loss = mdn.loss(X, Y, outcome)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update feature parameters using gradient descent
        for f in range(features):
            cur_mu, cur_sigma = feat_params[f]
            grad_mu = mdn.grad_mu_xi(X, Y, cur_mu, cur_sigma, f, outcome=2) / BATCH_SIZE
            grad_sigma = mdn.grad_sigma_xi(X, Y, cur_mu, cur_sigma, f, outcome=2) / BATCH_SIZE

            new_sigma = cur_sigma + (FEAT_PARAM_STEP * grad_sigma)

            # prevent 0 or negative standard deviation
            if new_sigma < MIN_SD:
                new_sigma = MIN_SD

            # Prevent zero or negative variance
            feat_params[f] = (cur_mu + (FEAT_PARAM_STEP * grad_mu), new_sigma)
            
        if batch % 100 == 0:
            logging.info(f"nn loss: {round(loss.item() / len(X), 4)}, time elapsed: {int(round((time.time()-start_time)/60, 0))} min [{(batch+1)*len(X):>5d}/{size:>5d}]")
            start_time = time.time()


# Evaluate neural network
@torch.no_grad()
def test_mdn(mdn, dataloader, outcome):
    size = len(dataloader.dataset)
    mdn.eval()

    start_time = time.time()

    test_loss = torch.zeros(1, 1).to(device)
    sq_error = torch.zeros(1, 1).to(device)
    for batch, sample in enumerate(dataloader):
        X = sample['features']
        Y = sample['outcomes']

        # Sent data to comptutation device
        X, Y = X.to(device), Y.to(device)

        # Calculate NN loss       
        loss = mdn.loss(X, Y, outcome)
        test_loss = test_loss + loss

        for m in range(len(X)):
            # Predict y
            y_probs = mdn.predict(X[m])
            max_y = 0
            max_p = 0
            for yp in y_probs:
                y = yp[0]
                p = yp[1]
                
                if p > max_p:
                    max_p = p
                    max_y = y
            # Compare to reference using square error
            sq_error = sq_error + (max_y - Y[m][outcome])**2

        if batch % 100 == 0:
            logging.info(f"test nn loss: {round(loss.item() / len(X), 3)}, sq_error: {round(sq_error.item() / len(X), 3)}, time elapsed: {int(round((time.time()-start_time)/60, 0))} min [{(batch+1)*len(X):>5d}/{size:>5d}]")
            start_time = time.time()

    test_loss = test_loss / size
    sq_error = sq_error / size

    return test_loss, sq_error

# Outcome index constants
P1_INDEX = 0
P25_INDEX = 1
P50_INDEX = 2
P75_INDEX = 3

# Retrain with full training dataset (selected 10)"
mdn = EMNet(features, 10, 2).double().to(device)
optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-4)

total_time = time.time()

init_feat_params(train_dataset.dataset)

for epoch in range(TRAIN_EPOCHS):
    logging.info(f"\nEpoch {epoch+1}/{TRAIN_EPOCHS}")
    train_mdn(mdn, train_dataloader, optimizer, P25_INDEX)

logging.info(f"Training completed in {(time.time()-total_time)/60:.{4}f}min")

logging.info("\nTesting model...")
test_loss, sq_error = test_mdn(mdn, test_dataloader, P25_INDEX)
logging.info("Testing complete:")
logging.info(f"Neural network loss: {round(test_loss.item(), 4)}, average squared error: {round(sq_error.item(), 4)}")

# Save the final model
torch.save(mdn, f'em_mdn.pt')
logging.info("Neural network saved to em_mdn.pt")

# Save feature distributions to disk
feat_dist_file = open('em_mdn_feat.txt', 'w')

for d in feat_params:
    feat_dist_file.write(f"{d[0].item()}, {d[1].item()}\n")

feat_dist_file.close()

logging.info("Feature distributions saved to em_mdn_feat.txt")

