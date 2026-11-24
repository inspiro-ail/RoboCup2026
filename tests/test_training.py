import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import train_lorax


class TrainingDatasetTests(unittest.TestCase):
    def test_deduplicates_without_changing_originals_and_retains_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, experiment, report = root / "voices", root / "experiment", root / "report"
            source.mkdir()
            experiment.mkdir()
            report.mkdir()
            rate = 40000
            t = np.arange(5 * rate) / rate
            audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            # The last second has a different pitch, so loss of the tail is detectable.
            audio[-rate:] = 0.2 * np.sin(2 * np.pi * 440 * t[-rate:])
            original = source / "voice.wav"
            sf.write(original, audio, rate, subtype="FLOAT")
            shutil.copy2(original, source / "voice duplicate.wav")
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            with patch.object(train_lorax, "REPORT_DIR", report):
                # This fixture is intentionally shorter than the production minimum of 10 segments.
                # Repeat clean utterances with pauses in a single unique source to exceed that minimum.
                long_audio = np.concatenate([np.concatenate([audio, np.zeros(rate // 2, dtype=np.float32)]) for _ in range(6)])
                sf.write(original, long_audio, rate, subtype="FLOAT")
                shutil.copy2(original, source / "voice duplicate.wav")
                before = {path.name: path.read_bytes() for path in source.iterdir()}
                manifest = train_lorax.prepare_dataset(source, experiment)
                reused = train_lorax.prepare_dataset(source, experiment)
            self.assertEqual(len(manifest["sources"]), 1)
            self.assertEqual(len(manifest["duplicates_skipped"]), 1)
            self.assertEqual(manifest, reused)
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})
            found_tail = False
            for key in manifest["chunks"]:
                waveform, output_rate = sf.read(experiment / "0_gt_wavs" / f"{key}.wav")
                self.assertEqual(output_rate, 40000)
                self.assertTrue(np.isfinite(waveform).all())
                self.assertLessEqual(len(waveform) / output_rate, 3.3)
                self.assertEqual(sf.info(str(experiment / "1_16k_wavs" / f"{key}.wav")).samplerate, 16000)
                tail = waveform[-min(len(waveform), output_rate // 2):]
                dominant = np.argmax(np.abs(np.fft.rfft(tail))) * output_rate / len(tail)
                found_tail |= abs(dominant - 440) < 5
            self.assertTrue(found_tail)
            # Prevent mixing old features/checkpoints with changed recordings.
            sf.write(source / "new.wav", audio / 2, rate, subtype="FLOAT")
            saved = (experiment / "dataset.json").read_bytes()
            with patch.object(train_lorax, "REPORT_DIR", report), self.assertRaisesRegex(ValueError, "dataset changed"):
                train_lorax.prepare_dataset(source, experiment)
            self.assertEqual(saved, (experiment / "dataset.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
