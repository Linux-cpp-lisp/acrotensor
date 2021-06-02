import importlib
from pathlib import Path

import torch

# From pytorch_scatter:
# https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/__init__.py

# suffix = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.ops.load_library(
    importlib.machinery.PathFinder().find_spec(
        "_acrotensor",
    ).origin
)
