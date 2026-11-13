"""Entry point for Streamlit (locally and on Streamlit Community Cloud).

    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Make src/ importable even where the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from screener.app.main import run  # noqa: E402

run()
