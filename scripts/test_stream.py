import argparse, os, subprocess, sys
p=argparse.ArgumentParser(); p.add_argument("--online",action="store_true"); p.add_argument("--work",default="work_stream"); a=p.parse_args()
if not a.online: raise SystemExit("Streaming test is opt-in. Rerun with --online; this may access gated data.")
if not os.getenv("HF_TOKEN"): print("HF_TOKEN is not set. Public sources may work; CulturaX requires approved access and a token.")
print("Accept the dataset terms on its Hugging Face page before CulturaX access: https://huggingface.co/datasets/uonlp/CulturaX")
raise SystemExit(subprocess.run([sys.executable,"hindi_llm.py","process","--source","hf","--online","--max_docs","3","--work",a.work],check=False).returncode)
