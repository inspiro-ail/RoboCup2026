"""Install the pinned official RVC inference backend and its shared models."""

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

BASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RVC_ROOT", BASE / "vendor" / "RVC")).resolve()
REPOSITORY = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
REVISION = "81eed5e8f68b6bed1789f682fe78cdd324495afc"
MODEL_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"


def run(*args, cwd=BASE):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def download(remote, target):
    if target.is_file() and target.stat().st_size:
        print(f"[OK] {target.relative_to(ROOT)}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"[Download] {remote}", flush=True)
    try:
        request = urllib.request.Request(MODEL_BASE + remote, headers={"User-Agent": "Lorax-RVC-Setup"})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-packages", action="store_true")
    args = parser.parse_args()
    if not (ROOT / "infer" / "cli.py").is_file():
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--depth", "1", REPOSITORY, ROOT)
        run("git", "fetch", "--depth", "1", "origin", REVISION, cwd=ROOT)
        run("git", "checkout", "--detach", REVISION, cwd=ROOT)
    if not args.skip_packages:
        if importlib.util.find_spec("torch") is None and shutil.which("nvidia-smi"):
            run(sys.executable, "-m", "pip", "install", "torch==2.5.1", "torchaudio==2.5.1",
                "--index-url", "https://download.pytorch.org/whl/cu121")
        run(sys.executable, "-m", "pip", "install", "-r", BASE / "requirements.txt")
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        download("hubert_base/" + name, ROOT / "assets" / "hubert_base" / name)
    download("rmvpe.pt", ROOT / "assets" / "rmvpe" / "rmvpe.pt")
    if os.name == "nt" and not shutil.which("ffmpeg"):
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            download(name, ROOT / name)
    # Use the same import path and environment as conversions, without loading a character model.
    env = os.environ.copy()
    env["RVC_CUDA_GRAPH"] = "0"
    env["RVC_AUDIO_FORCE_CPU"] = "1"
    subprocess.run([
        sys.executable, "-c",
        "from infer.vc.modules import VC; from infer.rmvpe import RMVPE; "
        "import torch; print('RVC imports OK; CUDA:', torch.cuda.is_available())",
    ], cwd=ROOT, env=env, check=True)
    for character in ("lorax", "pipsqueak"):
        (BASE / "models" / character).mkdir(parents=True, exist_ok=True)
    print("[OK] RVC engine installed. Character voices require trained models/<character>/voice.pth.")


if __name__ == "__main__":
    main()
