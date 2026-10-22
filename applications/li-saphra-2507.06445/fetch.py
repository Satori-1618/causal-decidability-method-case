"""Download the small, pinned public evidence files; never execute upstream code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REVISION = "dedb269eca6a9cf0bc409792003e6f815f2dce34"
BASE = "https://raw.githubusercontent.com/vli31/id-predict-ood/" + REVISION + "/"
PATHS = [
    "LICENSE", "README.md", "data/transformer_head_properties.csv",
    "heldout/results/mean_ablation_dyck.csv",
    "heldout/results/mean_ablation_dyck_single.csv",
    "heldout/mean_ablation_dyck.py", "heldout/mean_ablation_dyck_single.py",
    "execution/save_attn_activations.py", "execution/run_model_utils.py",
    "utils/minGPT/model.py", "utils/model.py", "utils/data.py",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch_one(path):
    target = ROOT / "upstream" / path
    if not target.exists():
        with urllib.request.urlopen(BASE + path, timeout=60) as response:
            data = response.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_bytes(data)
        temp.replace(target)
    data = target.read_bytes()
    return path, {"url": BASE + path, "bytes": len(data), "sha256": digest(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--establish-lock", action="store_true")
    args = parser.parse_args()
    lock_path = ROOT / "SOURCE_LOCK.json"
    if args.establish_lock and lock_path.exists():
        raise SystemExit("Source lock already exists; do not replace it.")
    if not args.establish_lock and not lock_path.exists():
        raise SystemExit("A source lock is required.")
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(fetch_one, PATHS))
    actual = {"repository": "https://github.com/vli31/id-predict-ood",
              "revision": REVISION, "files": files}
    if args.establish_lock:
        lock_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    elif actual != json.loads(lock_path.read_text()):
        raise SystemExit("Pinned source verification failed.")
    print(f"Verified {len(files)} pinned files ({sum(x['bytes'] for x in files.values()):,} bytes).")


if __name__ == "__main__":
    main()
