# launch.py — starts your Streamlit app from an .exe
import os, sys
from streamlit.web import cli as stcli

BASE = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else __file__)
os.environ.setdefault("HF_HOME", os.path.join(BASE, "models_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(BASE, "models_cache"))
os.environ.setdefault("TORCH_HOME", os.path.join(BASE, "torch_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

APP_PATH = os.path.join(BASE, "app.py")

sys.argv = [
    "streamlit", "run", APP_PATH,
    "--server.headless=true",
    "--server.port=8501",
    "--browser.gatherUsageStats=false",
]
sys.exit(stcli.main())
