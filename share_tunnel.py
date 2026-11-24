import os
import sys
import re
import time
import subprocess
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CLOUDFLARED_BIN = BASE_DIR / "bin" / "cloudflared.exe"

def is_server_running():
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/status",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False

def start_server_if_needed():
    if is_server_running():
        print("[OK] Local AI server is already active on port 8000.")
        return None
    print("[1/2] Starting local AI server (FastAPI + RVC)...")
    python_exe = BASE_DIR / "venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)
    server_proc = subprocess.Popen(
        [str(python_exe), "server.py"],
        cwd=str(BASE_DIR),
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    )
    for _ in range(30):
        time.sleep(1)
        if is_server_running():
            print("[OK] Server started successfully!")
            return server_proc
    return server_proc

def copy_to_clipboard(text: str):
    try:
        p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
        p.communicate(text.encode("utf-8"))
    except Exception:
        pass

def main():
    print("\n" + "=" * 65)
    print("  * The Lorax AI Voice Studio - Public Online Tunnel *")
    print("=" * 65 + "\n")

    if not CLOUDFLARED_BIN.exists():
        print(f"[ERROR] cloudflared.exe not found at: {CLOUDFLARED_BIN}")
        input("Press Enter to exit...")
        return

    start_server_if_needed()

    print("\n[2/2] Connecting Cloudflare Tunnel to create public HTTPS link...")
    print("Please wait a few seconds...\n")

    cmd = [str(CLOUDFLARED_BIN), "tunnel", "--url", "http://127.0.0.1:8000"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    tunnel_url = None
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for line in proc.stdout:
        match = pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break
        if "ERR" in line:
            print("Notice:", line.strip())

    if tunnel_url:
        copy_to_clipboard(tunnel_url)
        (BASE_DIR / "public_url.txt").write_text(tunnel_url, encoding="utf-8")

        print("\n" + "=" * 65)
        print("  * YOUR PUBLIC HTTPS LINK IS READY! *")
        print("=" * 65)
        print(f"\n  >>  {tunnel_url}  <<\n")
        print("  [x] Link has been automatically copied to your clipboard!")
        print("  [x] Saved to: public_url.txt")
        print("  [x] HTTPS is active: microphone works on iOS/Android/PC!")
        print("=" * 65 + "\n")
        print("Opening link in your default browser...\n")
        try:
            webbrowser.open(tunnel_url)
        except Exception:
            pass

        print("Keep this window OPEN as long as you want the link to stay active.")
        print("To stop sharing, press Ctrl+C or close this window.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nTunnel closed.")

if __name__ == "__main__":
    main()

