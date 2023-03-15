import torch

class EMNet(torch.nn.Module):
    def __init__(self, features=22, hidden_dim=10, out_dim=2):
        super(EMNet, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(features, hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.ReLU()
       	)

def convert_state_dict(model_pt_path):
    model = torch.load(f"../{model_pt_path}")
    torch.save(model.state_dict(), f"../trained/{model_pt_path}")


convert_state_dict("50p_em_mdn.pt")
convert_state_dict("25p_em_mdn.pt")
convert_state_dict("75p_em_mdn.pt")
