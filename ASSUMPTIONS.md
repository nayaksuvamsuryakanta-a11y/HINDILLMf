# Assumptions

- The local fixture is the default source and the only source enabled by default. Hugging Face sources require both config enablement and the explicit `--online` option.
- A small fixture may produce no validation document under the required every-100th-document default. The bin validator warns, and tiny training falls back to training tokens for diagnostic loss until validation data exists. The offline smoke test uses a smaller validation interval to exercise that path.
- Resume state is based on per-source processed-document counts. In-memory exact deduplication does not survive restarts, so a resumed run may encounter a duplicate already emitted before the restart.
- The `medium` Llama dimensions are chosen to approximate 170M parameters; exact count depends on the Transformers implementation and vocabulary size.
- The CLI is designed for current Python 3.10+ and compatible PyTorch/Transformers releases; dependency versions remain unpinned to allow Kaggle compatibility updates.
- The Week 1 notebook enables only `culturax_hi` for the requested `--source hf` sampling command. This assumes CulturaX access has been approved for the Kaggle account; its source is disabled in the repository config by default.
