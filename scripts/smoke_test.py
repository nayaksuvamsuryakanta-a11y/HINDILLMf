import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument("--offline",action="store_true"); p.add_argument("--cleanup",action="store_true"); a=p.parse_args()
work=ROOT/"work_debug"
def run(*args): subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
if not (ROOT/"tests/fixtures/sample_hi.txt").exists(): run("scripts/make_fixture.py")
run("hindi_llm.py","tokenizer","--source","local","--dry_run","--work",work)
run("hindi_llm.py","process","--source","local","--work",work,"--max_docs","20","--val_every","5","--overwrite")
run("scripts/validate_bins.py","--work",work)
(work/"ckpt.pt").unlink(missing_ok=True)
run("hindi_llm.py","train","--size","tiny","--device","cpu","--work",work,"--max_steps","2","--block","64","--micro_bs","1","--accum","1","--eval_every","1","--allow_train_eval","--max_minutes","5")
if a.cleanup:
    import shutil
    shutil.rmtree(work,ignore_errors=True)
print("Offline smoke test completed.")
