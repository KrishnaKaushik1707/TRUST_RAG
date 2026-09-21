"""
Test runner for scratch/notebooks/ directory.
Executes the full hybrid retrieval evaluation suite.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "backend"))
sys.path.insert(0, str(root_dir / "scratch"))

from test_retrieval_demo import run_retrieval_evaluation

if __name__ == "__main__":
    run_retrieval_evaluation()
