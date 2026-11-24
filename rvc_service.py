"""Direct speech-to-speech conversion using the official RVC inference CLI."""

import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
MODEL_DIR = BASE_DIR / "models"
RVC_ROOT = Path(os.environ.get("RVC_ROOT", BASE_DIR / "vendor" / "RVC")).resolve()
MAX_AUDIO_SECONDS = 30
logger = logging.getLogger("RVCService")

CHARACTERS = {
    "lorax": {"name": "The Lorax", "avatar": "lorax"},
    "pipsqueak": {"name": "Pipsqueak", "avatar": "pipsqueak"},
}


class RVCNotReadyError(RuntimeError):
    pass


class RVCService:
    def __init__(self):
        self.lock = Lock()
        self._hardware = None

    @staticmethod
    def get_training_status(character_id):
        report_path = BASE_DIR / "training" / character_id / "status.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        stage = report.get("stage")
        active = stage in {"preparing_dataset", "downloading_training_models", "extracting_rmvpe_f0",
                           "extracting_hubert_features", "building_retrieval_index", "training", "checking_conversion"}
        result = {"active": active, "stage": stage, "target_epochs": report.get("target_epochs")}
        if stage == "training":
            experiment = report.get("experiment", "")
            if re.fullmatch(r"[A-Za-z0-9_]+", experiment):
                try:
                    with (RVC_ROOT / "logs" / experiment / "train.log").open("rb") as log:
                        log.seek(max(0, log.seek(0, 2) - 16384))
                        tail = log.read().decode("utf-8", errors="replace")
                    epochs = re.findall(r"Training epoch:\s*(\d+)", tail)
                    if epochs:
                        result["current_epoch"] = int(epochs[-1])
                except OSError:
                    pass
        return result

    def get_character_model(self, character_id):
        if character_id not in CHARACTERS:
            raise ValueError("Неизвестный персонаж.")
        folder = MODEL_DIR / character_id
        model = folder / "voice.pth"
        if not model.is_file():
            candidates = sorted(folder.glob("*.pth"))
            if len(candidates) > 1:
                raise ValueError(f"В {folder} несколько моделей. Назовите выбранную voice.pth.")
            model = candidates[0] if candidates else None
        indices = sorted(p for p in folder.glob("*.index") if "trained" not in p.name.lower())
        preferred = folder / "voice.index"
        index = preferred if preferred.is_file() else indices[0] if len(indices) == 1 else None
        return model, index

    def runtime_missing(self):
        missing = []
        for relative in (
            "infer/cli.py", "assets/hubert_base/config.json",
            "assets/hubert_base/preprocessor_config.json",
            "assets/hubert_base/pytorch_model.bin", "assets/rmvpe/rmvpe.pt",
        ):
            if not (RVC_ROOT / relative).is_file():
                missing.append(relative)
        for module in ("torch", "torchaudio", "av", "soundfile", "numpy", "librosa",
                       "faiss", "parselmouth", "transformers", "ffmpeg"):
            if importlib.util.find_spec(module) is None:
                missing.append(f"Python: {module}")
        import shutil
        if not (RVC_ROOT / "ffmpeg.exe").is_file() and not shutil.which("ffmpeg"):
            missing.append("FFmpeg")
        return missing

    def get_system_status(self):
        if self._hardware is None:
            try:
                import torch
                cuda = torch.cuda.is_available()
                self._hardware = {
                    "cuda_available": cuda, "device": "cuda:0" if cuda else "cpu",
                    "gpu_name": torch.cuda.get_device_name(0) if cuda else "CPU",
                }
            except (ImportError, OSError):
                self._hardware = {"cuda_available": False, "device": "cpu", "gpu_name": "CPU"}
        missing = self.runtime_missing()
        characters = {}
        for character_id, conf in CHARACTERS.items():
            training = self.get_training_status(character_id)
            try:
                model, index = self.get_character_model(character_id)
                error = None
            except ValueError as exc:
                model, index, error = None, None, str(exc)
            characters[character_id] = {
                **conf, "available": bool(model) and not missing,
                "has_model": bool(model), "has_index": bool(index),
                "training": training,
                "reason": error or ("Нужна обученная RVC-модель voice.pth" if not model
                                     else "Запустите setup_rvc.ps1" if missing else None),
            }
        return {
            "status": "online", "app_name": "Lorax AI Voice Studio", "engine": "RVC",
            "hardware": self._hardware, "characters": characters,
            "runtime": {"ready": not missing, "missing": missing},
            "f0_method": "rmvpe", "preserves_timing": True,
        }

    @staticmethod
    def validate_pitch_model(model):
        import torch
        try:
            checkpoint = torch.load(str(model), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("Не удалось прочитать RVC-модель. Нужен экспортированный inference-файл .pth.") from exc
        if not isinstance(checkpoint, dict) or not {"config", "weight"} <= checkpoint.keys():
            raise ValueError("Нужна inference-модель RVC, а не тренировочный checkpoint.")
        if checkpoint.get("f0", 1) != 1:
            raise ValueError("Эта RVC-модель обучена без F0. Для переноса интонации нужна модель с F0 / pitch guidance.")

    @staticmethod
    def decode_audio(source, destination):
        """Decode browser audio without trimming pauses or changing speed."""
        import av
        import numpy as np
        import soundfile as sf

        chunks = []
        samples = 0
        try:
            with av.open(str(source)) as container:
                if not container.streams.audio:
                    raise ValueError("В файле нет аудиодорожки.")
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
                for frame in container.decode(audio=0):
                    for converted in resampler.resample(frame):
                        chunk = converted.to_ndarray().reshape(-1)
                        samples += chunk.size
                        if samples > MAX_AUDIO_SECONDS * 16000:
                            raise ValueError("Максимальная длительность записи — 30 секунд.")
                        chunks.append(chunk)
                for converted in resampler.resample(None):
                    chunk = converted.to_ndarray().reshape(-1)
                    samples += chunk.size
                    chunks.append(chunk)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Не удалось прочитать запись. Используйте WAV, WebM, OGG или MP4.") from exc
        if samples > MAX_AUDIO_SECONDS * 16000:
            raise ValueError("Максимальная длительность записи — 30 секунд.")
        if samples < 1600:
            raise ValueError("Запись слишком короткая. Запишите хотя бы одну фразу.")
        data = np.concatenate(chunks)
        if not np.isfinite(data).all() or np.max(np.abs(data)) < 1e-5:
            raise ValueError("В записи тишина. Проверьте микрофон.")
        sf.write(str(destination), data, 16000, subtype="PCM_16")

    def convert_voice(self, character_id, input_audio_path, output_wav_path,
                      pitch_shift=0, index_rate=0.75, protect=0.33, rms_mix_rate=0.0):
        model, index = self.get_character_model(character_id)
        if not model:
            training = self.get_training_status(character_id)
            if training and training["active"]:
                raise RVCNotReadyError("RVC-модель персонажа сейчас обучается. Она подключится автоматически после завершения.")
            raise RVCNotReadyError(
                f"Нет обученной RVC-модели: models/{character_id}/voice.pth. "
                "WAV-образец не заменяет модель. Инструкция: README.md."
            )
        missing = self.runtime_missing()
        if missing:
            raise RVCNotReadyError("RVC ещё не настроен. Запустите setup_rvc.ps1. Не хватает: " + ", ".join(missing))
        effective_index_rate = index_rate if index else 0.0
        output = Path(output_wav_path).resolve()
        (BASE_DIR / "temp").mkdir(exist_ok=True)
        with self.lock, TemporaryDirectory(prefix="rvc_", dir=BASE_DIR / "temp") as work:
            self.validate_pitch_model(model)
            wav = Path(work) / "input.wav"
            self.decode_audio(input_audio_path, wav)
            command = [
                sys.executable, "-m", "infer.cli", "--model", str(model.resolve()),
                "--input", str(wav), "--output", str(output), "--pitch", str(pitch_shift),
                "--f0-method", "rmvpe", "--index-rate", str(effective_index_rate),
                "--rms-mix-rate", str(rms_mix_rate), "--protect", str(protect),
                "--format", "wav",
            ]
            if index and effective_index_rate:
                command += ["--index", str(index.resolve())]
            env = os.environ.copy()
            env["PATH"] = str(RVC_ROOT) + os.pathsep + env.get("PATH", "")
            env["PYTHONIOENCODING"] = "utf-8"
            env["RVC_CUDA_GRAPH"] = "0"
            env["RVC_AUDIO_FORCE_CPU"] = "1"
            try:
                result = subprocess.run(command, cwd=RVC_ROOT, env=env, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=300,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode != 0:
                    logger.error("RVC failed:\n%s\n%s", result.stdout, result.stderr)
                    raise RuntimeError("Ошибка RVC. Подробности в консоли сервера; проверьте модель и установку движка.")
                import soundfile as sf
                if not output.is_file() or sf.info(str(output)).frames == 0:
                    raise RuntimeError("RVC не создал аудиозапись.")
            except subprocess.TimeoutExpired as exc:
                output.unlink(missing_ok=True)
                raise RuntimeError("RVC не завершил обработку за 5 минут. Попробуйте более короткую запись.") from exc
            except Exception:
                output.unlink(missing_ok=True)
                raise
        return {"engine": "RVC", "f0_method": "rmvpe", "pitch_shift": pitch_shift,
                "index_rate": effective_index_rate, "preserves_timing": True}

rvc_service = RVCService()