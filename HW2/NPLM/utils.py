# utils.py
import torch

def save_model(model, path: str):
    torch.save(model.state_dict(), path)

def load_model(model_class, path: str, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model
