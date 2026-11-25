# Local runbook

These commands use only local files unless `--online` is explicitly added.

```sh
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_env.py
python scripts/make_fixture.py
python hindi_llm.py tokenizer --source local --dry_run --work work_debug
python hindi_llm.py process --source local --work work_debug --max_docs 20
python scripts/validate_bins.py --work work_debug
python hindi_llm.py train --size tiny --device cpu --work work_debug --max_steps 2 --block 64 --micro_bs 1 --accum 1 --eval_every 1 --max_minutes 5
python scripts/smoke_test.py --offline
```

The smoke script uses `--val_every 5` and `--allow_train_eval` so tiny data can exercise training and evaluation paths. It only accepts the default `work_debug/` directory and refuses any other `--work` path, so its checkpoint deletion cannot target production artifacts. Keep generated artifacts there. Use `--cleanup` only when you want the smoke script to remove that directory. Normal processing defaults to reserving each 100th kept document for validation. Re-run the same `process` command to resume; committed token counts in schema-version-2 `manifest.json` control binary reconciliation. Extra binary tails are truncated, but missing committed bytes abort. Use `--overwrite` to reset both binary files and rebuild. Training checkpoints save every `--save_every` steps (defaulting to `--eval_every`) and at the final step; they validate data and tokenizer identity on resume. Checkpoints written before identity metadata was added cannot be resumed: the guard aborts with a checkpoint mismatch message. Start a fresh work directory to train again; do not bypass the guard. `--ignore_data_mismatch` exists, but it is dangerous and must not be used to force-resume across changed data or tokenizer artifacts.

Run `python scripts/test_resume.py` for offline resume, fingerprint, and evaluation-guard regression checks. It keeps its temporary directory unless `--cleanup` is explicitly supplied. Tested package versions and Kaggle dependency guidance are in [ENVIRONMENT.md](ENVIRONMENT.md).

For streaming, use `python scripts/test_stream.py --online` only after setting up access and accepting any gated dataset terms. It is not part of offline validation.
