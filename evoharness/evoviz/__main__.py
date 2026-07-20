"""python -m evoharness.evoviz <run_dir> [-o report.html]"""

import argparse
from pathlib import Path

from .report import generate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
parser.add_argument("-o", "--out", type=Path, default=None)
args = parser.parse_args()
out = generate(args.run_dir, args.out)
print(out)
