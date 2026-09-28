"""Smoke test: runs the quantized streaming tflite over raw wav clips and
reports max wake word probability per clip (ESPHome-style: trigger when the
probability stays above the cutoff for 5 consecutive inferences)."""
import random
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from pathlib import Path

from microwakeword.audio.audio_utils import generate_features_for_clip

MODEL = sys.argv[1] if len(sys.argv) > 1 else "shishok.tflite"
CUTOFF = 0.94
WINDOW = 5
STEP = 3  # frames per inference

random.seed(0)
interpreter = Interpreter(model_path=MODEL, num_threads=1)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
scale, zp = inp["quantization"]


def max_probability(wav_path):
    import librosa

    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    audio = (np.clip(audio, -1, 1) * 32768).astype(np.int16)
    feats = generate_features_for_clip(audio, step_ms=10)
    n = (feats.shape[0] - STEP) // STEP
    probs, above = [], 0
    best = 0.0
    for i in range(n):
        window = feats[i * STEP : i * STEP + STEP]
        q = np.round(window / scale + zp).astype(np.int8)
        interpreter.set_tensor(inp["index"], q.reshape(1, STEP, 40))
        interpreter.invoke()
        p = float(interpreter.get_tensor(out["index"])[0, 0]) * out["quantization"][0]
        best = max(best, p)
        above = above + 1 if p >= CUTOFF else 0
        if above >= WINDOW:
            return best, True
    return best, False


def run_set(name, directory, pattern, limit, pick=None):
    files = sorted(Path(directory).glob(pattern))
    if pick:
        files = [f for f in files if pick(f)]
    random.shuffle(files)
    files = files[:limit]
    triggered, results = 0, []
    for f in files:
        r = max_probability(str(f))
        if r is None:
            continue
        best, hit = r
        triggered += hit
        results.append(best)
        print(f"  {f.name:45s} max_p={best:.3f} {'TRIGGER' if hit else '-'}")
    print(
        f"{name}: {triggered}/{len(results)} triggered, "
        f"median max_p={np.median(results):.3f}\n"
    )
    return triggered, len(results), results


if __name__ == "__main__":
    print(f"cutoff={CUTOFF}, window={WINDOW}\n")
    run_set("POSITIVES (шишок)", "positives_raw", "*.wav", 20)
    run_set("ADVERSARIAL (мешок, шишка...)", "adversarial_raw", "*.wav", 15)
    run_set("NEG RU SPEECH (Common Voice)", "cv_ru_wav", "*.wav", 15)
    run_set("BACKGROUND (ESC-50)", "esc50_16k", "*.wav", 10)
    run_set("MUSIC (FMA)", "fma_16k", "*.wav", 5)
