import os
from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def safe_savefig(path, bbox_inches="tight"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches=bbox_inches)
    plt.close()