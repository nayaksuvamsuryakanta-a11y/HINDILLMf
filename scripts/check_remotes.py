"""Verify Hugging Face authentication and access to the configured model repo."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_remotes() -> bool:
    try:
        from dotenv import load_dotenv
        from huggingface_hub import HfApi, login
    except ImportError as exc:
        print(f"ERROR: Missing {exc.name}. Install dependencies with: pip install -r requirements.txt")
        return False

    load_dotenv(ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN", "").strip()
    repo_id = os.getenv("HF_REPO_ID", "").strip()
    if not token:
        print("ERROR: HF_TOKEN is missing. Set it in .env locally or as a Kaggle Secret.")
        return False
    if not repo_id:
        print("ERROR: HF_REPO_ID is missing. Set it in .env or the environment (for example, user/model).")
        return False

    try:
        login(token=token, add_to_git_credential=False)
    except (ConnectionError, TimeoutError, OSError) as exc:
        print(f"ERROR: Could not reach Hugging Face while validating HF_TOKEN: {exc}")
        return False
    except Exception as exc:
        status = getattr(exc, "response", None)
        status = getattr(status, "status_code", None)
        if status in (401, 403):
            print("ERROR: HF_TOKEN was rejected. Check that it is valid and not expired.")
        elif status is not None:
            print(f"ERROR: Hugging Face rejected login (HTTP {status}): {exc}")
        else:
            print(f"ERROR: Hugging Face login failed ({type(exc).__name__}): {exc}")
        return False

    try:
        HfApi().repo_info(repo_id=repo_id, repo_type="model", token=token)
    except (ConnectionError, TimeoutError, OSError) as exc:
        print(f"ERROR: Could not reach Hugging Face while checking {repo_id}: {exc}")
        return False
    except Exception as exc:
        status = getattr(exc, "response", None)
        status = getattr(status, "status_code", None)
        if status == 401:
            print(f"ERROR: HF_TOKEN cannot access model repository {repo_id} (unauthorized).")
        elif status == 403:
            print(f"ERROR: Access to model repository {repo_id} is forbidden; check token permissions.")
        elif status == 404:
            print(f"ERROR: Model repository {repo_id} was not found or is not visible to this token.")
        elif status is not None:
            print(f"ERROR: Repository access check failed for {repo_id} (HTTP {status}): {exc}")
        else:
            print(f"ERROR: Repository access check failed for {repo_id} ({type(exc).__name__}): {exc}")
        return False

    print(f"READY FOR KAGGLE: token is valid and {repo_id} is accessible.")
    print("Note: repo_info verifies authenticated read access, not write permission; ensure HF_TOKEN has write scope before uploading.")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if check_remotes() else 1)
