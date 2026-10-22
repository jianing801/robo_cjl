"""Test this workspace's RSL-RL checkout, not an unrelated editable install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))
