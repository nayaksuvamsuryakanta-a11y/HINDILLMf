#!/usr/bin/env python3
"""Offline-first Hindi language-model data and training pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "tests" / "fixtures" / "sample_hi.txt"
LOG = logging.getLogger("hindi_llm")
EOS_ID = 2


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_docs(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    yield from (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip())


def sources_config() -> list[dict]:
    p = ROOT / "configs" / "sources.json"
    return json.loads(p.read_text(encoding="utf-8"))["sources"]


def selected_sources(which: str, online: bool = False) -> list[dict]:
    raw = json.loads((ROOT / "configs" / "sources.json").read_text(encoding="utf-8"))
    online = online or bool(raw.get("online_enabled", False))
    configs = raw["sources"]
    local = [s for s in configs if s.get("enabled") and s["name"] == "local_fixture"]
    hf = [s for s in configs if s.get("enabled") and s.get("dataset")]
    if which == "local": return local
    if which == "hf":
        if not online: raise RuntimeError("Hugging Face sources require --online (and possibly HF_TOKEN).")
        return hf
    return local + (hf if online else [])


def documents(source: dict, online: bool = False) -> Iterator[str]:
    if source["name"] == "local_fixture":
        p = Path(source.get("path", "tests/fixtures/sample_hi.txt"))
        yield from read_docs(p if p.is_absolute() else ROOT / p)
        return
    if not online: raise RuntimeError("Online source requested without explicit --online.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing datasets. Install with: pip install -r requirements.txt") from exc
    stream = load_dataset(source["dataset"], source.get("config"), split="train", streaming=True)
    for row in stream:
        value = row.get("text") or row.get("content") or ""
        if value: yield str(value)


def clean(text: str, min_chars: int = 20, max_chars: int = 5000, devanagari_ratio: float = .6,
          latin_ratio: float = .25) -> str | None:
    s = unicodedata.normalize("NFC", " ".join(text.split()))
    if len(s) < min_chars or len(s) > max_chars or re.search(r"(?:https?://|www\.)\S+", s, re.I): return None
    letters = [c for c in s if c.isalpha()]
    if not letters: return None
    dev = sum("\u0900" <= c <= "\u097f" for c in letters)
    lat = sum("a" <= c.lower() <= "z" for c in letters)
    if dev / len(letters) < devanagari_ratio or lat / len(letters) > latin_ratio: return None
    punct = sum(unicodedata.category(c).startswith("P") for c in s)
    if punct / max(len(s), 1) > .18: return None
    words = s.split()
    if len(words) >= 5:
        grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if len(set(grams)) / len(grams) < .45: return None
    return s


def require_sentencepiece():
    try: import sentencepiece as spm
    except ImportError as exc: raise RuntimeError("Missing sentencepiece. Install with: pip install -r requirements.txt") from exc
    return spm


def tokenizer_cmd(a) -> None:
    spm = require_sentencepiece()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    online = a.online or bool(json.loads((ROOT / "configs" / "sources.json").read_text(encoding="utf-8")).get("online_enabled", False))
    candidates = selected_sources(a.source, online)
    lines: list[str] = []
    for src in candidates:
        for doc in documents(src, online):
            normalized = clean(doc, min_chars=1, devanagari_ratio=0, latin_ratio=1)
            if normalized:
                lines.extend(normalized.splitlines())
            if len(lines) >= a.max_sample_lines: break
    if not lines: raise RuntimeError("No usable local text sample found.")
    sample = work / "tokenizer_sample.txt"
    sample.write_text("\n".join(lines[:a.max_sample_lines]), encoding="utf-8")
    vocab = a.vocab
    if a.dry_run:
        vocab = min(vocab, max(32, len(set("".join(lines))) + 8))
    model_prefix = work / "hindi_bpe"
    spm.SentencePieceTrainer.train(input=str(sample), model_prefix=str(model_prefix), vocab_size=vocab,
        model_type="bpe", character_coverage=1.0, pad_id=0, unk_id=1, bos_id=-1, eos_id=EOS_ID,
        hard_vocab_limit=False, input_sentence_size=min(len(lines), a.max_sample_lines) if len(lines) > 100 else 0, shuffle_input_sentence=False)
    model = model_prefix.with_suffix(".model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    atomic_json(work / "tokenizer_meta.json", {"tokenizer_hash": digest, "vocab_size": spm.SentencePieceProcessor(model_file=str(model)).get_piece_size(), "eos_id": EOS_ID})
    proc = spm.SentencePieceProcessor(model_file=str(model))
    words = 0; pieces = 0
    for line in lines[:100]:
        words += len(line.split()); pieces += len(proc.encode(line, out_type=int))
    LOG.info("Tokenizer ready: vocab=%d, fertility=%.3f pieces/word", proc.get_piece_size(), pieces / max(words, 1))


def process_cmd(a) -> None:
    import numpy as np
    spm = require_sentencepiece()
    work = Path(a.work); model_path = work / "hindi_bpe.model"; meta_path = work / "tokenizer_meta.json"
    if not model_path.exists() or not meta_path.exists(): raise RuntimeError("Tokenizer missing; run tokenizer first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8")); fingerprint = meta["tokenizer_hash"]
    train_path, val_path, manifest_path = work / "train.bin", work / "val.bin", work / "manifest.json"
    manifest_exists = manifest_path.exists()
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_exists else {}
    if manifest_exists and not a.overwrite:
        required = ("schema_version", "train_tokens", "val_tokens", "tokenizer_hash")
        if any(k not in old for k in required):
            raise RuntimeError("Manifest is missing schema_version or token counts; reprocess with --overwrite.")
        if old["tokenizer_hash"] != fingerprint:
            raise RuntimeError("Tokenizer fingerprint changed while bins exist; pass --overwrite to rebuild.")
        for path, count, label in ((train_path, old["train_tokens"], "train"), (val_path, old["val_tokens"], "validation")):
            expected_bytes = int(count) * 2
            actual_bytes = path.stat().st_size if path.exists() else 0
            if actual_bytes < expected_bytes:
                raise RuntimeError(f"Data loss detected: {label} binary has {actual_bytes} bytes but manifest requires {expected_bytes}; restore it or reprocess with --overwrite.")
            discarded = (actual_bytes - expected_bytes) // 2
            with path.open("r+b" if path.exists() else "wb") as handle:
                handle.truncate(expected_bytes)
            if discarded:
                LOG.info("Discarded %d uncommitted %s tokens while reconciling binary with manifest", discarded, label)
    if a.overwrite:
        for p in (train_path, val_path):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
        manifest_path.unlink(missing_ok=True)
        old = {}
    # Keep the expected split files present even when a tiny local run has not
    # reached the first validation document (every 100th kept document).
    train_path.touch(exist_ok=True); val_path.touch(exist_ok=True)
    manifest = old or {"schema_version": 2, "source_counts": {}, "global_doc_idx": 0, "train_tokens": 0, "val_tokens": 0,
                       "tokenizer_hash": fingerprint, "last_updated": None, "sources": []}
    online = a.online or bool(json.loads((ROOT / "configs" / "sources.json").read_text(encoding="utf-8")).get("online_enabled", False))
    sources = selected_sources(a.source, online)
    if not old: manifest["sources"] = [s["name"] for s in sources]
    # In-memory line dedupe only; it does not persist across restarted sessions.
    seen: set[bytes] = set(); proc = spm.SentencePieceProcessor(model_file=str(model_path))
    started = time.monotonic(); processed = 0
    handles = {train_path: train_path.open("ab"), val_path: val_path.open("ab")}
    def flush_and_save() -> None:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        save_manifest(manifest_path, manifest)
    try:
        for src in sources:
            name = src["name"]; skip = int(manifest["source_counts"].get(name, 0))
            if name not in manifest["source_counts"]: manifest["source_counts"][name] = 0
            if name != "local_fixture" and online:
                LOG.warning("Streaming order may be non-deterministic; verify dataset state before relying on resume.")
            for doc in documents(src, online):
                if skip:
                    skip -= 1; continue
                manifest["source_counts"][name] += 1
                processed += 1
                item = clean(doc, a.min_chars, a.max_chars, a.min_devanagari_ratio, a.max_latin_ratio)
                if not item:
                    if processed % 1000 == 0: flush_and_save()
                    if (a.max_docs and processed >= a.max_docs) or (a.max_minutes and time.monotonic() - started >= a.max_minutes * 60): raise StopIteration
                    continue
                key = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
                if key in seen: continue
                seen.add(key)
                ids = proc.encode(item, out_type=int) + [EOS_ID]
                dest = val_path if (manifest["global_doc_idx"] + 1) % a.val_every == 0 else train_path
                np.asarray(ids, dtype=np.uint16).tofile(handles[dest])
                if dest == val_path: manifest["val_tokens"] += len(ids)
                else: manifest["train_tokens"] += len(ids)
                manifest["global_doc_idx"] += 1
                if processed % 1000 == 0: flush_and_save()
                if (a.max_docs and processed >= a.max_docs) or (a.max_minutes and time.monotonic() - started >= a.max_minutes * 60): raise StopIteration
    except StopIteration:
        LOG.info("Stopping at configured limit; progress saved.")
    except KeyboardInterrupt:
        LOG.info("Interrupted; saving processing progress.")
    finally:
        flush_and_save()
        for handle in handles.values():
            handle.close()


def save_manifest(path: Path, data: dict) -> None:
    data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(path, data)


def model_dims(size: str) -> tuple[int, int, int, int]:
    return {"tiny": (128, 2, 4, 256), "small": (512, 8, 8, 1376), "medium": (768, 21, 12, 2048)}[size]


def train_cmd(a) -> None:
    import numpy as np
    a.save_every = a.save_every or a.eval_every
    try:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc: raise RuntimeError("Missing torch/transformers. Install with: pip install -r requirements.txt") from exc
    work=Path(a.work); meta=json.loads((work/"tokenizer_meta.json").read_text())
    vocab=a.vocab or meta["vocab_size"]; tr=work/"train.bin"; va=work/"val.bin"
    if not tr.exists() or not va.exists(): raise RuntimeError("train.bin/val.bin missing; run process first.")
    train=np.memmap(tr,dtype=np.uint16,mode="r")
    manifest=json.loads((work/"manifest.json").read_text(encoding="utf-8"))
    val=np.memmap(va,dtype=np.uint16,mode="r") if va.stat().st_size else train
    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device)
    d_model,layers,heads,hidden=model_dims(a.size)
    cfg=LlamaConfig(vocab_size=vocab,hidden_size=d_model,intermediate_size=hidden,num_hidden_layers=layers,num_attention_heads=heads,num_key_value_heads=heads,max_position_embeddings=a.block,bos_token_id=1,eos_token_id=EOS_ID,pad_token_id=0,tie_word_embeddings=True)
    model=LlamaForCausalLM(cfg).to(device); model.train()
    decay=[]; no_decay=[]
    for _,p in model.named_parameters(): (decay if p.dim()>=2 else no_decay).append(p)
    opt=torch.optim.AdamW([{"params":decay,"weight_decay":.1},{"params":no_decay,"weight_decay":0.}],lr=a.lr)
    ckpt=work/"ckpt.pt"; step=0; last_saved_step=-1; scaler=None; use_fp16=False
    dimensions={"hidden_size":d_model,"num_hidden_layers":layers,"num_attention_heads":heads,"intermediate_size":hidden}
    train_tokens=int(manifest["train_tokens"]); val_tokens=int(manifest["val_tokens"])
    fingerprint=hashlib.blake2b(f"{meta['tokenizer_hash']}:{train_tokens}:{val_tokens}".encode(),digest_size=32).hexdigest()
    identity={"tokenizer_hash":meta["tokenizer_hash"],"vocab_size":vocab,"train_tokens":train_tokens,"val_tokens":val_tokens,"block":a.block,"size":a.size,"model_dimensions":dimensions,"manifest_schema_version":manifest.get("schema_version"),"data_fingerprint":fingerprint}
    if device == "cuda":
        major=torch.cuda.get_device_capability()[0]; use_fp16=major<8
        if use_fp16: scaler=torch.amp.GradScaler("cuda")
    if ckpt.exists():
        state=torch.load(ckpt,map_location=device,weights_only=False)
        mismatches=[f"{key}: expected {identity.get(key)!r}, checkpoint has {state.get(key)!r}" for key in identity if state.get(key)!=identity[key]]
        if mismatches and not a.ignore_data_mismatch:
            raise RuntimeError("Checkpoint data/config mismatch; " + "; ".join(mismatches) + ". Pass --ignore_data_mismatch only if you understand the risk.")
        if mismatches: LOG.warning("Ignoring checkpoint data/config mismatch (dangerous): %s", "; ".join(mismatches))
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); step=state["step"]
        if scaler and state.get("scaler"): scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"])
        if state.get("python_rng") is not None: random.setstate(state["python_rng"])
        if state.get("numpy_rng") is not None: np.random.set_state(state["numpy_rng"])
        if device == "cuda" and state.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(state["cuda_rng"])
        last_saved_step=step
        LOG.info("Resumed checkpoint at step %d",step)
    eval_data = val if len(val) > a.block + 1 else train
    if len(val) <= a.block + 1:
        if not a.allow_train_eval:
            raise RuntimeError("Validation split has fewer than block+1 tokens; provide more validation data or pass --allow_train_eval for diagnostic training-data evaluation.")
        LOG.warning("Validation split is too small; reported validation loss is computed on training data because --allow_train_eval was passed.")
    def batch(data):
        if len(data)<=a.block+1: raise RuntimeError("Binary split too short for selected block size.")
        ix=np.random.randint(0,len(data)-a.block-1,size=a.micro_bs)
        x=np.stack([data[i:i+a.block] for i in ix]).astype(np.int64); y=np.stack([data[i+1:i+a.block+1] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)
    def evaluate():
        model.eval(); losses=[]
        with torch.no_grad():
            for _ in range(5):
                x,y=batch(eval_data); losses.append(float(model(input_ids=x,labels=y).loss))
        model.train(); return sum(losses)/len(losses)
    start=time.monotonic()
    def save_checkpoint() -> None:
        nonlocal last_saved_step
        state={"model":model.state_dict(),"model_config":cfg.to_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict() if scaler else None,"step":step,"cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else None,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),**identity}
        tmp=ckpt.with_suffix(".pt.tmp"); began=time.monotonic(); torch.save(state,tmp); os.replace(tmp,ckpt)
        last_saved_step=step
        LOG.info("Checkpoint saved at step %d (%.2f MB, %.2f seconds)",step,ckpt.stat().st_size/(1024*1024),time.monotonic()-began)
    while step<a.max_steps and time.monotonic()-start<a.max_minutes*60:
        opt.zero_grad(set_to_none=True); loss_total=0.
        for _ in range(a.accum):
            x,y=batch(train)
            with torch.autocast(device_type=device, dtype=torch.float16 if use_fp16 else torch.bfloat16, enabled=device=="cuda"):
                loss=model(input_ids=x,labels=y).loss/a.accum
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            loss_total+=float(loss.detach())
        if scaler: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if step < a.warmup_steps:
            lr=a.lr*(step+1)/max(a.warmup_steps,1)
        else:
            progress=min(max((step-a.warmup_steps)/max(a.max_steps-a.warmup_steps,1),0.0),1.0)
            lr=a.lr*.5*(1.0+math.cos(math.pi*progress))
        for group in opt.param_groups: group["lr"]=lr
        if scaler: scaler.step(opt); scaler.update()
        else: opt.step()
        step+=1
        if step%a.eval_every==0: LOG.info("step=%d train_loss=%.4f val_loss=%.4f",step,loss_total,evaluate())
        if step%a.save_every==0: save_checkpoint()
    if last_saved_step != step:
        save_checkpoint()
    LOG.info("Training stopped at step %d; checkpoint %s",step,ckpt)
    if a.hf_repo:
        try:
            from huggingface_hub import HfApi
            if os.getenv("HF_TOKEN"): HfApi().upload_file(path_or_fileobj=str(ckpt),path_in_repo=ckpt.name,repo_id=a.hf_repo,repo_type="model")
            else: LOG.warning("HF_TOKEN absent; skipped optional Hub upload.")
        except Exception as e: LOG.warning("Optional Hub upload skipped: %s",e)


def load_for_inference(work: Path, checkpoint: Path | None = None):
    import torch
    from transformers import LlamaConfig,LlamaForCausalLM
    path=checkpoint or work/"ckpt.pt"
    state=torch.load(path,map_location="cpu",weights_only=False)
    cfg=LlamaConfig.from_dict(state["model_config"]) if isinstance(state,dict) and "model_config" in state else LlamaConfig(vocab_size=json.loads((work/"tokenizer_meta.json").read_text())["vocab_size"],hidden_size=128,intermediate_size=256,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4)
    model=LlamaForCausalLM(cfg); model.load_state_dict(state.get("model",state)); model.eval()
    return torch,model


def eval_cmd(a) -> None:
    import numpy as np
    val_path=Path(a.work)/"val.bin"
    if not val_path.exists() or val_path.stat().st_size//2 < a.block+1:
        raise RuntimeError(f"Validation split has fewer than block+1 ({a.block+1}) tokens; provide a larger validation split before evaluation.")
    torch,model=load_for_inference(Path(a.work),Path(a.checkpoint) if a.checkpoint else None)
    data=np.memmap(val_path,dtype=np.uint16,mode="r"); losses=[]
    with torch.no_grad():
        for _ in range(a.batches):
            i=random.randrange(0,len(data)-a.block-1); x=torch.tensor(data[i:i+a.block].astype("int64")).unsqueeze(0); y=torch.tensor(data[i+1:i+a.block+1].astype("int64")).unsqueeze(0); losses.append(float(model(input_ids=x,labels=y).loss))
    loss=sum(losses)/len(losses); result={"validation_loss":loss,"perplexity":math.exp(min(loss,80)),"approximate":False}
    atomic_json(Path(a.work)/"eval_metrics.json",result); print(json.dumps(result,indent=2))


def generate_cmd(a) -> None:
    import torch
    spm=require_sentencepiece(); work=Path(a.work); path=work/"ckpt.pt"
    if not path.exists(): raise RuntimeError(f"Checkpoint not found: {path}. Run train first.")
    torch,model=load_for_inference(work); proc=spm.SentencePieceProcessor(model_file=str(work/"hindi_bpe.model")); ids=proc.encode(a.prompt,out_type=int)
    x=torch.tensor([ids],dtype=torch.long); out=model.generate(x,max_new_tokens=a.max_new_tokens,do_sample=a.temperature>0,temperature=max(a.temperature,.01),top_p=a.top_p,repetition_penalty=a.repetition_penalty,eos_token_id=EOS_ID)
    print(proc.decode(out[0].tolist()))


def make_fixture() -> None:
    FIXTURE.parent.mkdir(parents=True,exist_ok=True)
    FIXTURE.write_text("नमस्ते दुनिया। यह हिंदी भाषा के लिए एक छोटा परीक्षण वाक्य है।\n\nभारत में अनेक भाषाएँ बोली जाती हैं और हिंदी उनमें प्रमुख है।\n\nयह एक छोटा वाक्य।\n\nhttps://example.com यह लिंक परीक्षण के लिए है।\n\nThis is an English heavy example for filtering.\n\nबारिश के बाद खेतों में हरियाली दिखाई देती है और किसान प्रसन्न होते हैं।\n\nबारिश बारिश बारिश बारिश बारिश बारिश बारिश बारिश बारिश बारिश बारिश बारिश।\n\nपुस्तकालय में विद्यार्थी पढ़ते हैं और नए विषयों के बारे में सीखते हैं।\n\nस्वच्छ जल और स्वस्थ पर्यावरण जीवन के लिए बहुत आवश्यक हैं।\n",encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("fixture"); f.set_defaults(func=lambda a:make_fixture())
    t=sub.add_parser("tokenizer"); t.add_argument("--source",choices=["local","hf","all"],default="local"); t.add_argument("--work",default="work"); t.add_argument("--vocab",type=int,default=32000); t.add_argument("--max_sample_lines",type=int,default=100000); t.add_argument("--dry_run",action="store_true"); t.add_argument("--online",action="store_true"); t.set_defaults(func=tokenizer_cmd)
    q=sub.add_parser("process"); q.add_argument("--source",choices=["local","hf","all"],default="local"); q.add_argument("--work",default="work"); q.add_argument("--online",action="store_true"); q.add_argument("--max_docs",type=int,default=0); q.add_argument("--max_minutes",type=float,default=0); q.add_argument("--chunk_docs",type=int,default=1000); q.add_argument("--min_chars",type=int,default=20); q.add_argument("--max_chars",type=int,default=5000); q.add_argument("--min_devanagari_ratio",type=float,default=.6); q.add_argument("--max_latin_ratio",type=float,default=.25); q.add_argument("--val_every",type=int,default=100); q.add_argument("--overwrite",action="store_true"); q.set_defaults(func=process_cmd)
    tr=sub.add_parser("train"); tr.add_argument("--size",choices=["tiny","small","medium"],default="tiny"); tr.add_argument("--device",choices=["auto","cuda","cpu"],default="auto"); tr.add_argument("--work",default="work"); tr.add_argument("--vocab",type=int); tr.add_argument("--max_steps",type=int,default=1000); tr.add_argument("--block",type=int,default=512); tr.add_argument("--micro_bs",type=int,default=2); tr.add_argument("--accum",type=int,default=8); tr.add_argument("--eval_every",type=int,default=100); tr.add_argument("--save_every",type=int); tr.add_argument("--allow_train_eval",action="store_true"); tr.add_argument("--ignore_data_mismatch",action="store_true"); tr.add_argument("--max_minutes",type=float,default=700); tr.add_argument("--lr",type=float,default=3e-4); tr.add_argument("--warmup_steps",type=int,default=100); tr.add_argument("--hf_repo"); tr.set_defaults(func=train_cmd)
    ev=sub.add_parser("evaluate"); ev.add_argument("--work",default="work"); ev.add_argument("--checkpoint"); ev.add_argument("--block",type=int,default=128); ev.add_argument("--batches",type=int,default=20); ev.set_defaults(func=eval_cmd)
    g=sub.add_parser("generate"); g.add_argument("--work",default="work"); g.add_argument("--prompt",required=True); g.add_argument("--max_new_tokens",type=int,default=80); g.add_argument("--temperature",type=float,default=.8); g.add_argument("--top_p",type=float,default=.9); g.add_argument("--repetition_penalty",type=float,default=1.1); g.set_defaults(func=generate_cmd)
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    args=parser().parse_args()
    try: args.func(args); return 0
    except Exception as exc: LOG.error("%s",exc); return 1

if __name__ == "__main__": raise SystemExit(main())
