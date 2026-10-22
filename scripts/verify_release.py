#!/usr/bin/env python3
"""Check bundled evidence with existing verifiers; no model execution or downloads.

Requires NumPy, SciPy and a Git checkout with history (pip install -e '.[verify]').
The historical Tracr verifier writes a report, so it runs on a temporary data copy;
archived Li/Saphra runs are likewise unpacked only into temporary directories.
The run fails unless every check in EXPECTED_CHECKS ran and passed.
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
MIXING = "applications/gur-arieh-2510.06182"
TRACR = "applications/tracr"
SAPHRA = "applications/li-saphra-2507.06445"

EXPECTED_CHECKS = (
    "archived Makelov files unchanged",
    "Makelov Q1: 64-pair relative comparison",
    "Makelov round 2",
    "Makelov round 3A",
    "Makelov round 3B: qualification stop",
    "Mixing Mechanisms: concentration-profile confirmation",
    "CausaLab MCQA: declared prediction groups",
    "Tracr: recorded tensors and exact population boundary",
    "Li/Saphra retrospective: stored outputs reproduce from the released tables",
    "Li/Saphra retrospective: separate raw-data recount",
    "Li/Saphra native follow-up: archived run verified on a temporary copy",
    "Li/Saphra round 004: independent recount from the archive",
    "Public claims: every CLAIMS.json entry (download-based checks reported as skipped)",
)
PASSED = []


def run(label, *args):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    # -E -s -B: ignore environment and user site, no bytecode; unlike -I, keep each
    # script's own directory importable (the Li/Saphra scripts import siblings).
    result = subprocess.run([sys.executable, "-E", "-s", "-B", *map(str, args)],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        print(f"FAIL: {label}\n{result.stdout}{result.stderr}", file=sys.stderr)
        raise SystemExit(result.returncode)
    PASSED.append(label)
    print(f"PASS: {label}", flush=True)


def main():
    if any(importlib.util.find_spec(name) is None for name in ("numpy", "scipy")):
        raise SystemExit("NumPy and SciPy are needed. Install with: python -m pip install -e '.[verify]'")
    try:
        shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT,
                                 capture_output=True, text=True)
    except OSError:
        shallow = None
    if shallow is None or shallow.returncode or shallow.stdout.strip() != "false":
        raise SystemExit("Use a full Git clone, not a source ZIP or a --depth clone: "
                         "git clone https://github.com/Satori-1618/causal-decidability-method-case")
    run("archived Makelov files unchanged", "scripts/check_frozen_files.py")
    run("Makelov Q1: 64-pair relative comparison", "examples/confirmed_read_source.py")
    run("Makelov round 2", f"{MAKELOV}/scripts/check_query_route_records.py",
        "--results", f"{MAKELOV}/results/query_route_confirmation")
    run("Makelov round 3A", f"{MAKELOV}/scripts/check_donor_factor_512_records.py",
        "--results", f"{MAKELOV}/results/donor_factor_confirmation_512")
    run("Makelov round 3B: qualification stop", f"{MAKELOV}/scripts/check_role_baseline_records.py",
        f"{MAKELOV}/results/role_baseline_development", "--check-only")
    run("Mixing Mechanisms: concentration-profile confirmation", f"{MIXING}/scripts/check_mixing_round1_records.py",
        "--results", f"{MIXING}/results/confirmation")
    run("CausaLab MCQA: declared prediction groups", "examples/causal_preflight.py",
        "--config", "applications/goodfire-mcqa-preflight/results/combined_predictions.json",
        "--structure-only")
    with tempfile.TemporaryDirectory(prefix="causal-release-check-") as temp:
        copied_run = Path(temp) / "confirmation_001"
        shutil.copytree(ROOT / TRACR / "results/confirmation_001", copied_run)
        run("Tracr: recorded tensors and exact population boundary", "scripts/check_tracr_records_portable.py",
            copied_run, "--freeze", ROOT / TRACR / "CONFIRMATION_FREEZE.json")
    run("Li/Saphra retrospective: stored outputs reproduce from the released tables",
        f"{SAPHRA}/analyze.py", "--check")
    run("Li/Saphra retrospective: separate raw-data recount", f"{SAPHRA}/verify_independently.py")
    run("Li/Saphra native follow-up: archived run verified on a temporary copy",
        "scripts/describe_claim_inputs.py", "saphra-native")
    run("Li/Saphra round 004: independent recount from the archive",
        "scripts/describe_claim_inputs.py", "saphra-averaged004")
    run("Public claims: every CLAIMS.json entry (download-based checks reported as skipped)",
        "scripts/check_claims.py")
    if tuple(PASSED) != EXPECTED_CHECKS:
        raise SystemExit(f"Expected checks {list(EXPECTED_CHECKS)} but ran {PASSED}")
    print(f"All {len(EXPECTED_CHECKS)} expected checks passed.")
    print("Verified stored evidence only; no new model runs or general reliability claim.")


if __name__ == "__main__":
    main()
