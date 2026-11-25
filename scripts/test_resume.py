"""Offline regression checks for processing resume and training guards."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--cleanup", action="store_true", help="Remove the temporary work directory after the checks")
args = parser.parse_args()
work = Path(tempfile.mkdtemp(prefix="hindi_llm_resume_"))


def run(*parts: str, expect_success: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, *map(str, parts)], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    output = (result.stdout or "") + (result.stderr or "")
    if expect_success and result.returncode:
        raise AssertionError(f"Command failed ({result.returncode}): {parts}\n{output}")
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"Command unexpectedly succeeded: {parts}\n{output}")
    return result


try:
    production_work = work / "production_work"
    production_work.mkdir()
    protected_checkpoint = production_work / "ckpt.pt"
    protected_checkpoint.write_bytes(b"keep this checkpoint")
    smoke_guard = run("scripts/smoke_test.py", "--offline", "--work", str(production_work), expect_success=False)
    assert "must not be pointed at a production work directory" in smoke_guard.stdout + smoke_guard.stderr
    assert protected_checkpoint.read_bytes() == b"keep this checkpoint", "Smoke-test guard deleted a non-debug checkpoint"

    run("hindi_llm.py", "tokenizer", "--source", "local", "--dry_run", "--work", str(work))
    process_args = ("hindi_llm.py", "process", "--source", "local", "--work", str(work), "--max_docs", "100", "--val_every", "2")
    run(*process_args)
    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    snapshots = {name: (work / name).read_bytes() for name in ("train.bin", "val.bin")}
    (work / "train.bin").write_bytes(snapshots["train.bin"] + b"\xff\xff\xff\xff")
    run(*process_args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, count_key in (("train.bin", "train_tokens"), ("val.bin", "val_tokens")):
        assert (work / name).stat().st_size == manifest[count_key] * 2
        assert (work / name).read_bytes() == snapshots[name], f"Resume duplicated or changed tokens in {name}"

    train_args = ("hindi_llm.py", "train", "--size", "tiny", "--device", "cpu", "--work", str(work),
                  "--max_steps", "2", "--block", "8", "--micro_bs", "1", "--accum", "1",
                  "--eval_every", "1", "--max_minutes", "2")
    run(*train_args)
    checkpoint = __import__("torch").load(work / "ckpt.pt", map_location="cpu", weights_only=False)
    assert checkpoint["data_fingerprint"]
    manifest["val_tokens"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mismatch = run("hindi_llm.py", "train", "--size", "tiny", "--device", "cpu", "--work", str(work),
                   "--max_steps", "3", "--block", "8", "--micro_bs", "1", "--accum", "1",
                   "--eval_every", "1", "--max_minutes", "2", expect_success=False)
    assert "val_tokens" in mismatch.stdout + mismatch.stderr

    (work / "val.bin").write_bytes(b"")
    evaluation = run("hindi_llm.py", "evaluate", "--work", str(work), "--block", "8", expect_success=False)
    assert "Validation split has fewer than block+1" in evaluation.stdout + evaluation.stderr
    print(f"Offline resume regressions passed. Temporary artifacts: {work}")
finally:
    if args.cleanup:
        import shutil
        shutil.rmtree(work)
