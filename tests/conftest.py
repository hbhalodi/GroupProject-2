import sys
from pathlib import Path

# Make `golu` and `rag_server` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
