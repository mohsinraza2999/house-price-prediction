import yaml

def train_config(path="configs/training.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def inference_config(path="configs/inference.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def paths_config(path="configs/paths.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def data_config(path="configs/data.yaml"):
    with open(path,"r") as f:
        return yaml.safe_load(f)