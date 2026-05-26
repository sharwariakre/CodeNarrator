from pathlib import Path
import sys

# Make codenarrator/ importable when pytest runs from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
