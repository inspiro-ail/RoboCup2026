"""Run upstream RVC training with Windows/4 GB loader settings."""

import os
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    root = Path(os.environ["RVC_TRAIN_ROOT"]).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    import torch
    from torch.utils.data import DataLoader

    class LocalDataLoader(DataLoader):
        def __init__(self, *args, **kwargs):
            kwargs.update(num_workers=0, persistent_workers=False, prefetch_factor=None, pin_memory=False)
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = LocalDataLoader
    torch.set_num_threads(2)
    if os.environ.get("RVC_TRAIN_DISABLE_CUDNN") == "1":
        torch.backends.cudnn.enabled = False
    runpy.run_module("train.train", run_name="__main__")
