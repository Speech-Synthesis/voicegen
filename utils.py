import json
import os
import torch

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self.__dict__[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return HParams(**data)

def filter_phones(phone_string):
    """
    Standard phoneme filtering to ensure alignment consistency between MFA TextGrid
    and the filelist string.
    Drops silent phones ('sil'), merges consecutive spaces/silences, or maps 'spn'.
    """
    if not phone_string:
        return ""
    phones = phone_string.strip().split()
    filtered = []
    for ph in phones:
        ph_clean = ph.lower().strip()
        # Drop silent tokens
        if ph_clean in ["sil", "sp", "spn", ""]:
            continue
        filtered.append(ph_clean)
    return " ".join(filtered)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # support state_dict vs direct dict
    if "model" in checkpoint_dict:
        model.load_state_dict(checkpoint_dict["model"], strict=False)
    else:
        model.load_state_dict(checkpoint_dict, strict=False)
        
    if optimizer is not None and "optimizer" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        
    iteration = checkpoint_dict.get("iteration", 1)
    learning_rate = checkpoint_dict.get("learning_rate", None)
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    state_dict = model.state_dict()
    torch.save({
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "learning_rate": learning_rate
    }, checkpoint_path)
