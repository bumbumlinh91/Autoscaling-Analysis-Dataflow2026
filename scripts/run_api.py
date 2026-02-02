import os
import subprocess
import time
import webbrowser
from pathlib import Path

import requests

# ==============================
# 1) Xác định PROJECT ROOT
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

HOST = "127.0.0.1"
PORT = 8000
DOCS_URL = f"http://{HOST}:{PORT}/docs"
HEALTH_URL = f"http://{HOST}:{PORT}/health"

print("🚀 Starting FastAPI server...")
print(f"📂 Project root: {PROJECT_ROOT}")
print(f"🌐 Target: {DOCS_URL}")

# ==============================
# 2) Khởi động uvicorn (non-blocking)
# ==============================
cmd = [
    "python", "-m", "uvicorn",
    "src.api.app:app",
    "--host", HOST,
    "--port", str(PORT),
    "--reload",
]

proc = subprocess.Popen(cmd)

# ==============================
# 3) Chờ API sẵn sàng 
# ==============================
timeout_s = 20
start = time.time()
ready = False

while time.time() - start < timeout_s:
    try:
        r = requests.get(HEALTH_URL, timeout=1)
        if r.status_code == 200:
            ready = True
            break
    except requests.exceptions.RequestException:
        pass
    time.sleep(0.3)

if ready:
    print("✅ API is ready! Opening browser...")
    webbrowser.open(DOCS_URL)
else:
    print("⚠️ API not ready in time. You can open manually:")
    print(DOCS_URL)

# ==============================
# 4) Chờ
# ==============================
try:
    proc.wait()
except KeyboardInterrupt:
    print("\n🛑 Stopping server...")
    proc.terminate()