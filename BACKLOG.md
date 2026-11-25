# Backlog

## Code freeze

No further pipeline code changes are permitted until Milestone A produces a recorded fertility number.

## Deferred work

- Make the 1,000-document manifest checkpoint fire even when a duplicate reaches that counter boundary; duplicate short-circuits currently can skip that save opportunity.
- Update the direct tiny-training example in `docs/RUNBOOK.md` to pass `--allow_train_eval` or use a sufficiently large validation split; the current fixture may not satisfy its block length.

- Verify the chosen Llama dimensions and end-to-end memory use on a Kaggle T4 before the pilot run.
- Consider persistent deduplication only if duplicate rates justify its storage and complexity.
- Record final data source versions, licenses, and consent/access status before corpus processing.
- Tokenizer sampling has no manifest/cursor resume: every invocation restarts the HF stream from the beginning, holds the sample in memory, then overwrites `tokenizer_sample.txt`. Add an atomic sample checkpoint with source identity/revision, processed row or line count, partial sample artifact, and tokenizer/config fingerprint; resume only when the source order/revision is stable, otherwise clearly restart and warn.
