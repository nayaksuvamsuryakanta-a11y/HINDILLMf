# hindi-llm-from-scratch

An offline-first foundation for cleaning Hindi text, training a 32k SentencePiece tokenizer, packing resumable `uint16` token streams, and training a small causal-LM pilot. The target pilot is about 42M parameters, followed later by a 170M base model on a Kaggle T4. This repository does not download data or launch large training jobs by default.

## Local-safe workflow

Install dependencies once in a virtual environment, then run the CPU/offline workflow in [docs/RUNBOOK.md](docs/RUNBOOK.md). `scripts/smoke_test.py --offline` uses only the tiny fixture and `work_debug/`; it does not need a token, internet, or CUDA. Generated bins/checkpoints are ignored by Git. The fixture has noisy and clean examples, and can be regenerated with `python scripts/make_fixture.py`.

`python hindi_llm.py tokenizer --source local --dry_run --work work_debug` builds a small local tokenizer. `process` normalizes NFC, filters and deduplicates documents, appends EOS ID 2, and sends every 100th kept document to validation by default. Tiny runs can have an undersized validation split; diagnostic training evaluation then requires `--allow_train_eval`. Processing resumes from a schema-version-2 manifest: extra uncommitted binary tails are truncated, but missing committed bytes abort with a data-loss message. Deduplication uses an in-memory BLAKE2 set and therefore does not persist across restarts.

Commands: `fixture`, `tokenizer`, `process`, `train`, `evaluate`, and `generate`. For usage, run `python hindi_llm.py --help` and the relevant command with `--help`.

## Online and Kaggle-only operations

Hugging Face streaming is disabled unless `--online` is passed and the chosen source is enabled in `configs/sources.json`. Gated CulturaX access may require accepting dataset terms and setting `HF_TOKEN`; offline commands never require it. Online stream order may change, so verify dataset state before trusting a resumed stream.

Use Kaggle for full tokenizer samples, multi-session data processing, and pilot training. See [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md). Target assumptions: one T4, fp16 when required, 20 GB working disk, sessions under 12 hours, no FlashAttention-2 dependency, and no DDP. Keep chunks and samples bounded, track the manifest, and persist artifacts privately. `small` is the ~42M pilot configuration and `medium` the ~170M configuration; do not run either locally.

## Resume and storage

Processing appends to `train.bin` and `val.bin`, fsyncs them before atomically updating `manifest.json`, and reconciles their lengths on resume. Re-run the same process command to resume; use `--overwrite` to rebuild from empty bins. Training checkpoints are written atomically at `--save_every` intervals (default: `--eval_every`) and at the final step. Checkpoints are bound to tokenizer, data counts, and model settings. Checkpoints written before identity metadata was added cannot be resumed: the guard aborts with a mismatch message. Start a fresh work directory to train again; do not bypass the guard. `--ignore_data_mismatch` exists, but it is dangerous and must not be used to force-resume across changed data or tokenizer artifacts. See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for tested dependency versions and Kaggle guidance. Avoid retaining source copies and temporary tokenizer samples on the 20 GB Kaggle disk; save only the tokenizer, bins, manifest, and checkpoints you need.

Milestones are in [PLAN_EXECUTION.md](PLAN_EXECUTION.md), and open work is tracked in [BACKLOG.md](BACKLOG.md). Assumptions are recorded in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Phase 2: Kaggle Setup

The repository includes `.gitattributes` with `* text=auto eol=lf`, so Git normalizes text files to LF for Kaggle's Linux environment. Before using remote datasets or pushing checkpoints, configure `HF_TOKEN` and `HF_REPO_ID` in a local `.env` file (never commit it), or configure them as Kaggle Secrets. Verify credentials and repository access with:

```sh
python scripts/check_remotes.py
```

To include those checks in the environment report, run `python scripts/check_env.py --online`. These commands contact Hugging Face. The check logs in and reads the configured repository metadata; `repo_info` verifies authenticated repository access but does not independently prove write permission. The optional checkpoint upload will still require a token with write scope.

Follow [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md) for the Kaggle workflow and [PLAN_EXECUTION.md](PLAN_EXECUTION.md) for the Nov 4 execution schedule.

The Week 1 Kaggle tokenizer notebook is [kaggle/week1_tokenizer.ipynb](kaggle/week1_tokenizer.ipynb). It keeps Accelerator set to None through data sampling and tokenizer work; reserve the T4 for the later training session.
