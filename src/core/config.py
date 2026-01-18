import os
import yaml

def load_config():
    path = os.path.join(os.getcwd(), "config/settings.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
