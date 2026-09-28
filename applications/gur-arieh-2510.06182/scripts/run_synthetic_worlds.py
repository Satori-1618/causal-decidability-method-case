"""Run the six synthetic worlds (analyzer verification only; no model).

    python scripts/run_synthetic_worlds.py            # print compact outcomes
    python scripts/run_synthetic_worlds.py --write    # store results/synthetic_worlds/worlds.json
    python scripts/run_synthetic_worlds.py --check    # stored outcomes equal a fresh run
"""
import argparse
import json
import sys
from pathlib import Path

APPLICATION = Path(__file__).resolve().parents[1]
if str(APPLICATION / "src") not in sys.path:
    sys.path.insert(0, str(APPLICATION / "src"))
from mixing_synthetic_worlds import VERIFICATION_PARAMETERS, run_all  # noqa: E402

STORED = APPLICATION / "results/synthetic_worlds/worlds.json"


def render():
    return json.dumps({
        "label": "SYNTHETIC: analyzer verification, not evidence",
        "verification_parameters": VERIFICATION_PARAMETERS,
        "note": "These parameters exist only to exercise the analyzer. The values proposed "
                "for the real run are in PROPOSED_VALUES.json and await approval.",
        "worlds": run_all(),
    }, indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.write:
        STORED.parent.mkdir(parents=True, exist_ok=True)
        STORED.write_text(text)
        print(f"wrote {STORED.relative_to(APPLICATION)}")
    elif args.check:
        if json.loads(STORED.read_text()) != json.loads(text):
            parser.exit(1, "FAILED: stored synthetic-world outcomes differ from a fresh run\n")
        print("PASS: synthetic-world outcomes reproduce (analyzer verification, not evidence).")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
