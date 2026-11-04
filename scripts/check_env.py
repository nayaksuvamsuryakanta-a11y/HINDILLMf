from __future__ import annotations
import importlib.util, os, sys
import argparse
from pathlib import Path

required = ["torch", "transformers", "datasets", "sentencepiece", "numpy", "huggingface_hub"]
print("Hindi LLM environment readiness")
print(f"Python: {sys.version.split()[0]} {'OK' if sys.version_info >= (3,10) else 'NEEDS 3.10+'}")
missing=[]
for name in required:
    ok=importlib.util.find_spec(name) is not None
    print(f"{name}: {'available' if ok else 'missing'}")
    if not ok: missing.append(name)
cuda=False
if importlib.util.find_spec("torch"):
    import torch
    cuda=torch.cuda.is_available()
print(f"CUDA: {'available' if cuda else 'not available (CPU local smoke tests remain supported)'}")
print(f"HF_TOKEN: {'present' if os.getenv('HF_TOKEN') else 'not set (offline usage is unaffected)'}")
print("Gated dataset access: not checked online; accept CulturaX terms and configure HF_TOKEN on Kaggle if needed.")
if missing: print("Install missing dependencies with: pip install -r requirements.txt")

parser = argparse.ArgumentParser(description="Check local and optional remote readiness.")
parser.add_argument("--online", action="store_true", help="Also verify Hugging Face token and repository access.")
args = parser.parse_args()
if args.online:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_remotes import check_remotes
    if not check_remotes():
        raise SystemExit(1)
