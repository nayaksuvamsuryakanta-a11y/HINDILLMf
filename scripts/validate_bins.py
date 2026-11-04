import argparse, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument("--work",default="work"); a=p.parse_args(); work=Path(a.work)
meta_path=work/"tokenizer_meta.json"
if not meta_path.exists(): raise SystemExit("Missing tokenizer_meta.json")
vocab=json.loads(meta_path.read_text(encoding="utf-8"))["vocab_size"]
ok=True
for name in ("train.bin","val.bin"):
    path=work/name
    if not path.exists(): print(f"ERROR: {name} missing"); ok=False; continue
    if path.stat().st_size == 0:
        print(f"{name}: 0 bytes, 0 uint16 tokens, EOS=0, out_of_range=0")
        if name == "train.bin": ok=False
        else: print("WARNING: validation split is empty; every 100th kept document is reserved by default")
        continue
    ids=np.memmap(path,dtype=np.uint16,mode="r")
    invalid=int(np.count_nonzero(ids>=vocab)); eos=int(np.count_nonzero(ids==2))
    print(f"{name}: {path.stat().st_size:,} bytes, {len(ids):,} uint16 tokens, EOS={eos}, out_of_range={invalid}")
    if invalid or (eos==0 and name=="train.bin"): ok=False
    if name=="val.bin" and (len(ids)<100 or eos==0): print("WARNING: validation split is empty or very small; every 100th kept document is reserved by default")
if not ok: raise SystemExit(1)
