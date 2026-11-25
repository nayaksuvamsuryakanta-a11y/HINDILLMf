# Assumptions

- The local fixture is the default source and the only source enabled by default. Hugging Face sources require both config enablement and the explicit `--online` option.
- A small fixture may produce no usable validation split under the every-100th-kept-document default. Training requires `--allow_train_eval` to use training data for diagnostic evaluation; standalone evaluation never uses that fallback.
- Processing resumes from a schema-version-2 manifest and per-source processed-document counts. Manifest token counts define the committed binary prefixes; extra tail bytes are truncated on resume, while binaries shorter than the manifest are treated as data loss. In-memory exact deduplication does not survive restarted sessions.
- The `medium` Llama dimensions are chosen to approximate 170M parameters; exact count depends on the Transformers implementation and vocabulary size.
- The CLI is designed for Python 3.10+; tested package versions are recorded in `docs/ENVIRONMENT.md`. Kaggle's preinstalled CUDA stack should be preserved.
- Checkpoint identity uses tokenizer hash, manifest token counts/schema, model settings, and block size. `--ignore_data_mismatch` bypasses these checks only as a dangerous explicit recovery option.
- The Week 1 notebook enables only `culturax_hi` for the requested `--source hf` sampling command. This assumes CulturaX access has been approved for the Kaggle account; its source is disabled in the repository config by default.
- The smoke-test `--work` path must resolve exactly to this repository's `work_debug` directory; equivalent path spellings are accepted after resolution.
- The Kaggle environment report records missing package versions and unavailable CUDA status as `not installed`, and writes the same report values to `work/env_report.json`.
