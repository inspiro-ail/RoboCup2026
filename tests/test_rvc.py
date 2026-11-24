import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

import rvc_service as service_module
from rvc_service import RVCService
from server import app, TEMP_DIR


def audio_bytes(seconds=1, sample_rate=48000):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * 220 * t)
    signal[:int(0.2 * sample_rate)] = 0
    signal[-int(0.2 * sample_rate):] = 0
    stream = io.BytesIO()
    sf.write(stream, signal, sample_rate, format="WAV")
    return stream.getvalue()


class AudioTests(unittest.TestCase):
    def test_decode_preserves_timing_pauses_and_pitch(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "browser.audio", Path(folder) / "decoded.wav"
            source.write_bytes(audio_bytes())
            RVCService.decode_audio(source, target)
            data, rate = sf.read(target)
            self.assertEqual(rate, 16000)
            self.assertEqual(len(data), 16000)
            self.assertLess(np.max(np.abs(data[:3000])), 1e-4)
            self.assertLess(np.max(np.abs(data[-3000:])), 1e-4)
            voice = data[4000:12000]
            frequency = np.argmax(np.abs(np.fft.rfft(voice))) * rate / len(voice)
            self.assertAlmostEqual(frequency, 220, delta=2)

    def test_rejects_invalid_silent_and_overlong_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "input", Path(folder) / "out.wav"
            for raw in (b"invalid", audio_bytes(0.1), audio_bytes(30.1)):
                source.write_bytes(raw)
                with self.assertRaises(ValueError):
                    RVCService.decode_audio(source, target)


class RVCIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)
        self.model_patch = patch.object(service_module, "MODEL_DIR", self.root / "models")
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.addCleanup(self.work.cleanup)
        self.model = self.root / "models" / "lorax" / "voice.pth"
        self.model.parent.mkdir(parents=True)
        torch.save({"config": [], "weight": {}, "f0": 1}, self.model)
        self.source = self.root / "input.wav"
        self.source.write_bytes(audio_bytes())
        self.target = self.root / "output.wav"
        self.service = RVCService()

    def test_direct_rvc_parameters_and_optional_index(self):
        for has_index in (False, True):
            if has_index:
                self.model.with_suffix(".index").write_bytes(b"index fixture")
            def inference(command, **kwargs):
                self.assertEqual(command[1:3], ["-m", "infer.cli"])
                self.assertEqual(command[command.index("--f0-method") + 1], "rmvpe")
                self.assertEqual(command[command.index("--pitch") + 1], "0")
                self.assertEqual(command[command.index("--rms-mix-rate") + 1], "0.0")
                self.assertNotIn("--speed", command)
                sf.write(self.target, np.ones(16000) * 0.1, 16000)
                return subprocess.CompletedProcess(command, 0, "Success", "")
            with patch.object(self.service, "runtime_missing", return_value=[]), patch.object(service_module.subprocess, "run", side_effect=inference):
                result = self.service.convert_voice("lorax", self.source, self.target)
            self.assertEqual(result["index_rate"], 0.75 if has_index else 0)
            self.assertTrue(result["preserves_timing"])
            self.target.unlink()

    def test_inference_failure_removes_partial_output(self):
        def inference(command, **kwargs):
            self.target.write_bytes(b"partial wav")
            return subprocess.CompletedProcess(command, 1, "", "invalid model")
        with patch.object(self.service, "runtime_missing", return_value=[]), patch.object(service_module.subprocess, "run", side_effect=inference):
            with self.assertRaises(RuntimeError):
                self.service.convert_voice("lorax", self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_does_not_treat_training_index_as_retrieval_index(self):
        (self.model.parent / "trained_IVF.index").write_bytes(b"not an added index")
        self.assertIsNone(self.service.get_character_model("lorax")[1])

    def test_rejects_model_without_pitch_guidance(self):
        torch.save({"config": [], "weight": {}, "f0": 0}, self.model)
        with patch.object(self.service, "runtime_missing", return_value=[]), patch.object(service_module.subprocess, "run") as inference:
            with self.assertRaisesRegex(ValueError, "F0"):
                self.service.convert_voice("lorax", self.source, self.target)
        inference.assert_not_called()


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def post(self, **data):
        return self.client.post("/api/convert", files={"audio": ("recording.webm", audio_bytes(), "audio/webm")}, data=data)

    def test_missing_model_is_503_and_upload_is_cleaned(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(service_module, "MODEL_DIR", Path(folder)), patch.object(RVCService, "get_training_status", return_value=None):
            before = set(TEMP_DIR.glob("upload_*"))
            response = self.post(character="lorax")
            self.assertEqual(response.status_code, 503)
            self.assertIn("voice.pth", response.json()["detail"])
            self.assertEqual(set(TEMP_DIR.glob("upload_*")), before)

    def test_training_model_is_reported_without_enabling_conversion(self):
        training = {"active": True, "stage": "training", "current_epoch": 5, "target_epochs": 100}
        with tempfile.TemporaryDirectory() as folder, patch.object(service_module, "MODEL_DIR", Path(folder)), patch.object(RVCService, "get_training_status", return_value=training):
            status = self.client.get("/api/status").json()["characters"]["lorax"]
            self.assertFalse(status["available"])
            self.assertEqual(status["training"]["current_epoch"], 5)
            response = self.post(character="lorax")
            self.assertEqual(response.status_code, 503)
            self.assertIn("обучается", response.json()["detail"])

    def test_rejects_old_tts_options_and_invalid_parameters(self):
        for parameters in ({"translate_to_en": "true"}, {"custom_text": "new text"}, {"speed": "0.75"}):
            self.assertEqual(self.post(**parameters).status_code, 400)
        for parameters in ({"pitch_shift": "25"}, {"pitch_shift": "nan"}, {"index_rate": "1.5"}, {"rms_mix_rate": "nan"}):
            self.assertEqual(self.post(**parameters).status_code, 422)
        self.assertEqual(self.post(character="other").status_code, 400)

    def test_status_reports_rvc_and_missing_model(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(service_module, "MODEL_DIR", Path(folder)):
            result = self.client.get("/api/status").json()
            self.assertEqual(result["engine"], "RVC")
            self.assertFalse(result["characters"]["lorax"]["available"])
            self.assertFalse(result["characters"]["lorax"]["has_model"])

    def test_static_and_output_routes(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/images/lorax.jpg").status_code, 200)
        self.assertEqual(self.client.get("/outputs/" + "f" * 32 + ".wav").status_code, 404)
        self.assertEqual(self.client.get("/outputs/" + "g" * 32 + ".wav").status_code, 404)


if __name__ == "__main__":
    unittest.main()
