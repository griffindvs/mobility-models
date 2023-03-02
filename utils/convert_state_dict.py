import torch

def convert_state_dict(model_pt_path):
    sample = torch.rand(22)
    model = torch.load(f"../{model_pt_path}")
    torch.save(model.state_dict(), f"../trained/{model_pt_path}")


convert_state_dict("50p_em_mdn.pt")
convert_state_dict("25p_em_mdn.pt")
convert_state_dict("75p_em_mdn.pt")
