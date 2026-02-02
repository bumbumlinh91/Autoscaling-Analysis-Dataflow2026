import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

os.environ.setdefault("API_BASE", "http://127.0.0.1:8000")

dashboard_file = PROJECT_ROOT / "dashboard" / "dashboard.py"

cmd = [
    sys.executable, "-m", "streamlit",
    "run", str(dashboard_file),
]

print("🚀 Đang mở Streamlit dashboard ")

subprocess.Popen(
    cmd,
    creationflags=subprocess.CREATE_NEW_CONSOLE
)