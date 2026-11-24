"""Prepare the supplied Lorax recordings, train official RVC v2/F0, and install it."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from setup_rvc import BASE, ROOT, download

VOICES = {
    "lorax": {"source": "voices Lorax", "experiment": "lorax_v2_40k"},
    "pipsqueak": {"source": "voices Pipsquik", "experiment": "pipsqueak_v2_40k"},
}
VOICE = "lorax"
REPORT_DIR = BASE / "training" / VOICE


def state(stage, **details):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_at_unix": time.time(), **details}
    (REPORT_DIR / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{VOICE} RVC] {stage}", flush=True)


def environment():
    env = os.environ.copy()
    env["PATH"] = str(ROOT) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["RVC_CUDA_GRAPH"] = "0"
    env["RVC_AUDIO_FORCE_CPU"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["RVC_TRAIN_ROOT"] = str(ROOT)
    env["TORCH_CUDNN_V8_API_DISABLED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    return env


def run_module(module, *args):
    entry = [str(BASE / "rvc_train_worker.py")] if module == "train.train" else ["-m", module]
    subprocess.run([sys.executable, *entry, *map(str, args)], cwd=ROOT, env=environment(), check=True)


def prepare_dataset(source, experiment):
    import av
    import librosa
    import numpy as np
    import soundfile as sf
    from scipy import signal
    sys.path.insert(0, str(ROOT))
    from train.dataset.slicer2 import Slicer

    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"})
    if not paths:
        raise ValueError(f"No audio files found in {source}")
    seen, unique, duplicates = set(), [], []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            duplicates.append(path.name)
        else:
            seen.add(digest)
            unique.append((path, digest))
    fingerprint = hashlib.sha256("".join(sorted(seen)).encode()).hexdigest()
    manifest_path = experiment / "dataset.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("The source dataset changed. Use a new --experiment to preserve previous training.")
        expected = [experiment / sub / f"{key}.wav" for sub in ("0_gt_wavs", "1_16k_wavs") for key in manifest["chunks"]]
        if all(path.is_file() for path in expected):
            return manifest
    for sub in ("0_gt_wavs", "1_16k_wavs"):
        (experiment / sub).mkdir(parents=True, exist_ok=True)
    slicer = Slicer(sr=40000, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=300)
    bh, ah = signal.butter(5, 48, btype="high", fs=40000)
    manifest = {"fingerprint": fingerprint, "sources": [], "duplicates_skipped": duplicates, "chunks": [], "raw_seconds": 0.0, "training_seconds_with_overlap": 0.0}
    for source_index, (path, digest) in enumerate(unique):
        frames = []
        with av.open(str(path)) as container:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=40000)
            for frame in container.decode(audio=0):
                frames.extend(x.to_ndarray().reshape(-1) for x in resampler.resample(frame))
            frames.extend(x.to_ndarray().reshape(-1) for x in resampler.resample(None))
        audio = np.concatenate(frames).astype(np.float32)
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-4:
            raise ValueError(f"Silent or invalid source: {path.name}")
        duration = len(audio) / 40000
        manifest["sources"].append({"file": path.name, "sha256": digest, "seconds": duration})
        manifest["raw_seconds"] += duration
        audio = signal.lfilter(bh, ah, audio).astype(np.float32)
        chunk_index = 0
        for utterance in slicer.slice(audio):
            if len(utterance) < 1.2 * 40000:
                continue
            # Balanced segments keep every utterance's tail; overlap avoids cutting phonemes away.
            count = max(1, math.ceil(len(utterance) / (3.0 * 40000)))
            bounds = np.linspace(0, len(utterance), count + 1, dtype=int)
            for index in range(count):
                start = max(0, bounds[index] - (6000 if index else 0))
                end = min(len(utterance), bounds[index + 1] + (6000 if index < count - 1 else 0))
                chunk = utterance[start:end]
                peak = np.max(np.abs(chunk))
                if peak < 1e-4:
                    continue
                chunk = (0.675 * chunk / peak + 0.25 * chunk).astype(np.float32)
                key = f"{source_index:02d}_{chunk_index:04d}"
                sf.write(experiment / "0_gt_wavs" / f"{key}.wav", chunk, 40000, subtype="FLOAT")
                downsampled = librosa.resample(chunk, orig_sr=40000, target_sr=16000)
                sf.write(experiment / "1_16k_wavs" / f"{key}.wav", downsampled, 16000, subtype="FLOAT")
                manifest["chunks"].append(key)
                manifest["training_seconds_with_overlap"] += len(chunk) / 40000
                chunk_index += 1
        print(f"[Dataset] {path.name}: {duration:.1f}s, {chunk_index} segments", flush=True)
    if len(manifest["chunks"]) < 10:
        raise ValueError("Too few usable voice segments to train.")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(manifest_path, REPORT_DIR / "dataset.json")
    return manifest


def create_training_config(experiment, manifest):
    records = []
    for key in manifest["chunks"]:
        paths = [experiment / "0_gt_wavs" / f"{key}.wav", experiment / "3_feature768" / f"{key}.npy",
                 experiment / "2a_f0" / f"{key}.wav.npy", experiment / "2b-f0nsf" / f"{key}.wav.npy"]
        if not all(path.is_file() for path in paths):
            raise RuntimeError(f"Missing extracted features for segment {key}; inspect the extraction log.")
        records.append("|".join(path.as_posix() for path in paths) + "|0")
    mute = ROOT / "logs" / "mute"
    mute_paths = [mute / "0_gt_wavs/mute40k.wav", mute / "3_feature768/mute.npy", mute / "2a_f0/mute.wav.npy", mute / "2b-f0nsf/mute.wav.npy"]
    if not all(p.is_file() for p in mute_paths):
        raise RuntimeError("Official mute training samples are missing.")
    records += ["|".join(p.as_posix() for p in mute_paths) + "|0"] * 2
    random.Random(1234).shuffle(records)
    (experiment / "filelist.txt").write_text("\n".join(records), encoding="utf-8")
    config_file = experiment / "config.json"
    config = json.loads((config_file if config_file.exists() else ROOT / "configs/v1/40k.json").read_text(encoding="utf-8"))
    config["train"]["batch_size"] = 1
    config["train"]["log_interval"] = 10
    config["train"]["segment_size"] = 6400
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")


def install_and_preview(experiment_name, epochs):
    import torch
    final = ROOT / "assets/weights" / f"{experiment_name}.pth"
    if not final.is_file():
        raise RuntimeError("RVC training did not export the final model.")
    checkpoint = torch.load(str(final), map_location="cpu", weights_only=True)
    if checkpoint.get("f0") != 1 or checkpoint.get("version") != "v2":
        raise RuntimeError("Exported model is not RVC v2 with F0.")
    indices = sorted((ROOT / "logs" / experiment_name).glob("added_*.index"))
    if len(indices) != 1:
        raise RuntimeError("Expected exactly one trained added index.")
    target = BASE / "models" / VOICE
    target.mkdir(parents=True, exist_ok=True)
    for source, name in ((final, "voice.pth"), (indices[0], "voice.index")):
        temporary = target / (name + ".new")
        shutil.copy2(source, temporary)
        temporary.replace(target / name)
    preview = BASE / "outputs" / f"{uuid.uuid4().hex}.wav"
    state("checking_conversion", epochs=epochs)
    from rvc_service import rvc_service
    result = rvc_service.convert_voice(VOICE, BASE / "reference" / VOICE / "voice.wav", preview)
    import soundfile as sf
    info = sf.info(str(preview))
    state("ready", epochs=epochs, model=str(target / "voice.pth"), index=str(target / "voice.index"),
          preview=str(preview), preview_seconds=info.duration, conversion=result)
    print(f"[Ready] {VOICE} model installed. Preview: {preview}", flush=True)


def main():
    global VOICE, REPORT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", choices=sorted(VOICES), default="lorax")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    VOICE = args.voice
    REPORT_DIR = BASE / "training" / VOICE
    args.source = args.source or BASE / VOICES[VOICE]["source"]
    args.experiment = args.experiment or VOICES[VOICE]["experiment"]
    if not args.experiment.replace("_", "").isalnum() or not 1 <= args.epochs <= 1200:
        parser.error("Use a simple experiment name and 1–1200 epochs.")
    experiment = ROOT / "logs" / args.experiment
    experiment.mkdir(parents=True, exist_ok=True)
    state("preparing_dataset", experiment=args.experiment)
    manifest = prepare_dataset(args.source.resolve(), experiment)
    state("dataset_prepared", segments=len(manifest["chunks"]), raw_seconds=manifest["raw_seconds"])
    if args.prepare_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requires a compatible NVIDIA GPU and Torch installation.")
    state("downloading_training_models")
    for name in ("f0G40k.pth", "f0D40k.pth"):
        download("pretrained_v2/" + name, ROOT / "assets/pretrained_v2" / name)
    mute_zip = ROOT / ".model-downloads/mute.zip"
    download("mute.zip", mute_zip)
    with zipfile.ZipFile(mute_zip) as archive:
        # Extract only within the dedicated RVC logs directory.
        destination = (ROOT / "logs").resolve()
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError("Invalid path in mute training archive.")
        archive.extractall(destination)
    state("extracting_rmvpe_f0")
    run_module("train.dataset.extract_f0", "cuda", 1, 0, 0, experiment, "true")
    state("extracting_hubert_features")
    run_module("train.dataset.extract_hubert_feature", "cuda:0", 1, 0, 0, experiment, "v2", "true")
    create_training_config(experiment, manifest)
    state("building_retrieval_index")
    run_module("train.train_index", args.experiment, "v2", ROOT / "assets/indices", 2, "single")
    (ROOT / "assets/weights").mkdir(parents=True, exist_ok=True)
    state("training", experiment=args.experiment, target_epochs=args.epochs, batch_size=1,
          segments=len(manifest["chunks"]), log=str(experiment / "train.log"))
    run_module("train.train", "-e", args.experiment, "-sr", "40k", "-f0", 1, "-bs", 1, "-g", 0,
               "-te", args.epochs, "-se", 10, "-pg", "assets/pretrained_v2/f0G40k.pth",
               "-pd", "assets/pretrained_v2/f0D40k.pth", "-l", 1, "-c", 0, "-sw", 1, "-v", "v2")
    install_and_preview(args.experiment, args.epochs)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        state("failed", error=str(error))
        raise
