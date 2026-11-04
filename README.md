# hindi-llm-from-scratch

An offline-first foundation for cleaning Hindi text, training a 32k SentencePiece tokenizer, packing resumable `uint16` token streams, and training a small causal-LM pilot. The target pilot is about 42M parameters, followed later by a 170M base model on a Kaggle T4. This repository does not download data or launch large training jobs by default.

## Local-safe workflow

Install dependencies once in a virtual environment, then run the CPU/offline workflow in [docs/RUNBOOK.md](docs/RUNBOOK.md). `scripts/smoke_test.py --offline` uses only the tiny fixture and `work_debug/`; it does not need a token, internet, or CUDA. Generated bins/checkpoints are ignored by Git. The fixture has noisy and clean examples, and can be regenerated with `python scripts/make_fixture.py`.

`python hindi_llm.py tokenizer --source local --dry_run --work work_debug` builds a small local tokenizer. `process` normalizes NFC, filters and deduplicates documents, appends EOS ID 2, and sends every 100th kept document to validation by default. Tiny runs with fewer than 100 documents can have an empty validation split; the CPU smoke test uses `--val_every 5` to exercise both files. Deduplication uses an in-memory BLAKE2 set and therefore does not persist across restarts. Local resume skips already processed documents as counted in `manifest.json`.

Commands: `fixture`, `tokenizer`, `process`, `train`, `evaluate`, and `generate`. For usage, run `python hindi_llm.py --help` and the relevant command with `--help`.

## Online and Kaggle-only operations

Hugging Face streaming is disabled unless `--online` is passed and the chosen source is enabled in `configs/sources.json`. Gated CulturaX access may require accepting dataset terms and setting `HF_TOKEN`; offline commands never require it. Online stream order may change, so verify dataset state before trusting a resumed stream.

Use Kaggle for full tokenizer samples, multi-session data processing, and pilot training. See [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md). Target assumptions: one T4, fp16 when required, 20 GB working disk, sessions under 12 hours, no FlashAttention-2 dependency, and no DDP. Keep chunks and samples bounded, track the manifest, and persist artifacts privately. `small` is the ~42M pilot configuration and `medium` the ~170M configuration; do not run either locally.

## Resume and storage

Processing appends to `train.bin` and `val.bin` and atomically updates `manifest.json`. Re-run the same process command to resume. Change tokenizer only after backing up or removing old bins; a fingerprint mismatch aborts unless `--overwrite` is explicit. Training resumes from `work/ckpt.pt`, written atomically. Avoid retaining source copies and temporary tokenizer samples on the 20 GB Kaggle disk; save only the tokenizer, bins, manifest, and checkpoints you need.

Milestones are in [PLAN_EXECUTION.md](PLAN_EXECUTION.md), and open work is tracked in [BACKLOG.md](BACKLOG.md). Assumptions are recorded in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Phase 2: Kaggle Setup

The repository includes `.gitattributes` with `* text=auto eol=lf`, so Git normalizes text files to LF for Kaggle's Linux environment. Before using remote datasets or pushing checkpoints, configure `HF_TOKEN` and `HF_REPO_ID` in a local `.env` file (never commit it), or configure them as Kaggle Secrets. Verify credentials and repository access with:

```sh
python scripts/check_remotes.py
```

To include those checks in the environment report, run `python scripts/check_env.py --online`. These commands contact Hugging Face. The check logs in and reads the configured repository metadata; `repo_info` verifies authenticated repository access but does not independently prove write permission. The optional checkpoint upload will still require a token with write scope.

Follow [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md) for the Kaggle workflow and [PLAN_EXECUTION.md](PLAN_EXECUTION.md) for the Nov 4 execution schedule.
