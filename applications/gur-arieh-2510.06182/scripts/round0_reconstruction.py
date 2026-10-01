"""Round 0 (RETROSPECTIVE): print, write or check the reconstruction. No model is loaded.

    python scripts/round0_reconstruction.py            # print
    python scripts/round0_reconstruction.py --write    # store results/round0_retrospective/round0.json
    python scripts/round0_reconstruction.py --check    # stored file equals a fresh reconstruction
"""
import argparse
import json
import sys
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
REPOSITORY = APPLICATION.parents[1]
for path in (APPLICATION / "src", REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from mixing_round0 import reconstruct  # noqa: E402

STORED = APPLICATION / "results/round0_retrospective/round0.json"


def render(result):
    return json.dumps(result, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render(reconstruct())
    if args.write:
        STORED.parent.mkdir(parents=True, exist_ok=True)
        STORED.write_text(text)
        print(f"wrote {STORED.relative_to(REPOSITORY)}")
    elif args.check:
        if STORED.read_text() != text:
            parser.exit(1, "FAILED: stored Round 0 result differs from the reconstruction\n")
        print("PASS: stored Round 0 result equals the reconstruction (RETROSPECTIVE, no model).")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
