import torch
from torch.distributions import Normal
import numpy as np
import copy

class EMNet(torch.nn.Module):
    def __init__(self, features=22, hidden_dim=10, out_dim=2):
        super(EMNet, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(features, hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.ReLU()
       	)
    def forward(self, x):
        params = self.seq(x)
        mu, sigma = torch.tensor_split(params, params.shape[0], dim=0)

        return mu, (sigma+1)**2

SAMPLES = 100
EPSILON = 1e-3

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

features = ['hhinc_mean2000', 'mean_commutetime2000', 'frac_coll_plus2000', 'frac_coll_plus2010', 
            'med_hhinc1990', 'med_hhinc2016', 'popdensity2000', 'poor_share2010', 'poor_share2000', 
            'poor_share1990', 'gsmn_math_g3_2013', 'traveltime15_2010', 'emp2000', 'singleparent_share1990',
            'singleparent_share2010', 'singleparent_share2000', 'mail_return_rate2010', 'jobs_total_5mi_2015', 
            'jobs_highpay_5mi_2015', 'popdensity2010', 'ann_avg_job_growth_2004_2013', 'job_density_2013']

def inference(model, inp):
    mu_nn, sigma_nn = model.forward(torch.tensor(inp).to(device))
    return mu_nn.item(), sigma_nn.item()

def run_sa(model_pt_path):
    model = EMNet().to(device)
    model_state = torch.load(f"trained/{model_pt_path}", map_location=device)
    model.load_state_dict(model_state)
    model.eval()

    model_name = model_pt_path.split(".")[0]

    feat_dist_file = open(f"trained/{model_name}_feat.txt", 'r')
    feat_params = []

    for line in feat_dist_file.readlines():
        lineSplit = line.split(",")
        mu = float(lineSplit[0])
        sigma = float(lineSplit[1])

        feat_params.append((mu, sigma))
    feat_dist_file.close()

    partials = [0 for i in range(len(feat_params))]
    for i in range(1):
        # Generate random sample input vector
        sample = []
        for f in feat_params:
            mu, sigma = f
            gen = Normal(mu, sigma).sample()
            sample.append(gen)

        # For each feature, compute numeric estimation of partial derivative
        original = copy.deepcopy(sample)
        for j in range(len(feat_params)):
            # f(x+h)
            sample[j] = sample[j] + EPSILON
            fdx_p = inference(model, sample)
            # f(x-h)
            sample[j] = original[j] - EPSILON
            fdx_m = inference(model, sample)
            # Two point numeric differentiation
            df_mu = (fdx_p[0] - fdx_m[0]) / (2*EPSILON)

            partials[j] += df_mu

            # Reset sample
            sample[j] = original[j]

    # Average samples
    for p in range(len(partials)):
        partials[p] = partials[p] / SAMPLES

    print(f"{model_name} feature sensitivity (x10^5):")
    for p in range(len(partials)):
        print("{: >4} : {: >29} : {: >6}".format(p, features[p], round(partials[p]*(10**5), 4)))
    print("\n")

run_sa("25p_em_mdn.pt")
run_sa("50p_em_mdn.pt")
run_sa("75p_em_mdn.pt")
