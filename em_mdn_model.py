import torch

class EMNet(torch.nn.Module):
    def __init__(self, features, hidden_dim, out_dim):
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
