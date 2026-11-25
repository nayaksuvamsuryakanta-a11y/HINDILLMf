# Tested environment

The offline CPU workflow was checked with:

- Python 3.12
- torch 2.15.0+cpu
- transformers 5.22.1
- datasets 5.1.0
- sentencepiece 0.2.2
- numpy 2.5.4
- huggingface_hub 2.4.0
- python-dotenv 1.2.4
- tqdm 4.70.1 (installed in the environment, but unused by this repository)

The local PyTorch CPU wheel is installed with:

```powershell
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining packages with `python -m pip install -r requirements.txt`. `torch` is intentionally unpinned because local development uses the CPU wheel while Kaggle provides a CUDA build. `tqdm` was removed from `requirements.txt` because the project does not use it.

## Kaggle

First inspect Kaggle's preinstalled versions with `python scripts/check_env.py` and confirm the required modules are available. Do not reinstall or upgrade `torch`, `transformers`, `datasets`, or `numpy` on Kaggle; in particular, preserve Kaggle's CUDA-enabled PyTorch build. Install only missing pure-Python helper packages when necessary, as described in [KAGGLE_SETUP.md](KAGGLE_SETUP.md).
