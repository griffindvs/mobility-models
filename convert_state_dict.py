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

def convert_state_dict(model_pt_path):
    sample = torch.rand(22)
    model = torch.load(model_pt_path)
    torch.save(model.state_dict(), f"trained/{model_pt_path}")


convert_state_dict("50p_em_mdn.pt")
convert_state_dict("25p_em_mdn.pt")
convert_state_dict("75p_em_mdn.pt")
