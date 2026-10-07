from pathlib import Path
import yaml

def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop('extends', None)
    return ({**load_config(path.parent / parent), **cfg} if parent else cfg)
