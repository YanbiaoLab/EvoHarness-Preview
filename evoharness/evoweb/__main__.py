import argparse
from pathlib import Path

from .server import serve

parser = argparse.ArgumentParser(description="EvoHarness live console")
parser.add_argument("root", type=Path, help="directory containing run dirs")
parser.add_argument("--port", type=int, default=7861)
args = parser.parse_args()
serve(args.root, args.port)
