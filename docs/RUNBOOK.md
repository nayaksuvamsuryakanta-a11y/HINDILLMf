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

The smoke script uses `--val_every 5` internally so the tiny data exercises validation. Keep generated artifacts in `work_debug/`. Use `--cleanup` only when you want the smoke script to remove that directory. Normal processing defaults to reserving each 100th kept document for validation. Re-run the same `process` command to resume; processing state is saved to `manifest.json`. A changed tokenizer cannot reuse existing bins unless `--overwrite` is specified.

For streaming, use `python scripts/test_stream.py --online` only after setting up access and accepting any gated dataset terms. It is not part of offline validation.
