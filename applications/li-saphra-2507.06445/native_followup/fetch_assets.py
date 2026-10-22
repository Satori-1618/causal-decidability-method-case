"""Fetch the fixed development checkpoint and original evaluation inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REVISION = "dedb269eca6a9cf0bc409792003e6f815f2dce34"
PATHS = [
    "utils/minGPT/utils.py",
    "data/model_weights/run_1aez5d6p/run_1aez5d6p_checkpoint_5.pt",
    "data/model_preds/indist_data_preds.csv",
    "data/model_preds/ood_data_preds.csv",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--establish-lock", action="store_true")
    args = parser.parse_args()
    lock = ROOT / "ASSET_LOCK.json"
    if args.establish_lock and lock.exists():
        raise SystemExit("Refusing to overwrite asset lock")
    if not args.establish_lock and not lock.exists():
        raise SystemExit("Missing asset lock")
    expected = json.loads(lock.read_text()) if lock.exists() else None
    records = {}
    for name in PATHS:
        url = "https://raw.githubusercontent.com/vli31/id-predict-ood/" + REVISION + "/" + name
        path = ROOT / "cache" / name
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        record = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if expected and expected["files"][name] != record:
            raise SystemExit("Asset hash mismatch: " + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        records[name] = record
        print(name, record["bytes"], record["sha256"])
    if args.establish_lock:
        lock.write_text(json.dumps({"revision": REVISION, "files": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
