#!/usr/bin/env python3
"""Check bundled evidence with existing verifiers; no model execution or downloads.

Requires NumPy, SciPy and a Git checkout with history (pip install -e '.[verify]').
The historical Tracr verifier writes a report, so it runs on a temporary data copy.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAKELOV = "applications/makelov-2311.17030"
GOODFIRE = "applications/gur-arieh-2510.06182"
TRACR = "applications/tracr"


def run(label, *args):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-I", "-B", *map(str, args)],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        print(f"FAIL: {label}\n{result.stdout}{result.stderr}", file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f"PASS: {label}", flush=True)


def main():
    if any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy")):
        raise SystemExit("NumPy and SciPy are needed. Install with: python -m pip install -e '.[verify]'")
    if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                      capture_output=True).returncode:
        raise SystemExit("Use a Git clone with full history, not a source ZIP.")
    run("archived Makelov files unchanged", "scripts/check_frozen_files.py")
    run("Makelov Q1: 64-pair relative comparison", "examples/confirmed_read_source.py")
    run("Makelov round 2", f"{MAKELOV}/scripts/check_query_route_records.py",
        "--results", f"{MAKELOV}/results/query_route_confirmation")
    run("Makelov round 3A", f"{MAKELOV}/scripts/check_donor_factor_512_records.py",
        "--results", f"{MAKELOV}/results/donor_factor_confirmation_512")
    run("Makelov round 3B: qualification stop", f"{MAKELOV}/scripts/check_role_baseline_records.py",
        f"{MAKELOV}/results/role_baseline_development", "--check-only")
    run("Goodfire: concentration-profile confirmation", f"{GOODFIRE}/scripts/check_mixing_round1_records.py",
        "--results", f"{GOODFIRE}/results/confirmation")
    run("Goodfire MCQA: declared prediction groups", "examples/causal_preflight.py",
        "--config", "applications/goodfire-mcqa-preflight/results/combined_predictions.json",
        "--structure-only")
    with tempfile.TemporaryDirectory(prefix="causal-release-check-") as temp:
        copied_run = Path(temp) / "confirmation_001"
        shutil.copytree(ROOT / TRACR / "results/confirmation_001", copied_run)
        run("Tracr: recorded tensors and exact population boundary", "scripts/check_tracr_records_portable.py",
            copied_run, "--freeze", ROOT / TRACR / "CONFIRMATION_FREEZE.json")
    print("Verified stored evidence only; no new model runs or general reliability claim.")


if __name__ == "__main__":
    main()
