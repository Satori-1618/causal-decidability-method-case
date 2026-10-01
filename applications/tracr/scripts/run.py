#!/usr/bin/env python3
"""Run the Tracr application from any working directory."""
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracr_demo.experiment import main

if __name__ == "__main__":
    main()
