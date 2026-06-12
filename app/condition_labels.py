import json
import os

def load_condition_labels(base_dir: str):
    p = os.path.join(base_dir, "datasets", "symcat_400", "labels.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        except Exception:
            pass
    return {}