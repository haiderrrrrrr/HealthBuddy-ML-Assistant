import os
import json
import torch
try:
    from models.lifestyle_model import LifestyleNet
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.lifestyle_model import LifestyleNet

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def _resolve_latest_run_dir(base_name: str):
    root = os.path.join(BASE_DIR, "stored_models", base_name, "latest")
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    subdirs.sort()
    if not subdirs:
        raise FileNotFoundError(f"No runs found under {root}")
    return os.path.join(root, subdirs[-1])

def load_lifestyle_model():
    run_dir = _resolve_latest_run_dir("LifestyleRisk")
    model_path = os.path.join(run_dir, "lifestyle_risk_model.pt")
    metrics_path = os.path.join(run_dir, "metrics.json")

    with open(metrics_path, "r") as f:
        meta = json.load(f)

    in_features = meta["in_features"]

    model = LifestyleNet(in_features)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
