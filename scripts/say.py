"""Synthesizes speech from text with one of the Russian Piper voices.

Examples:
    ./.venv/bin/python scripts/say.py "Привет, мир!"
    ./.venv/bin/python scripts/say.py "Шишок." --voice irina --out shishok.wav
    ./.venv/bin/python scripts/say.py "Шишок." --voice voices/ru_RU-dmitri-medium.onnx --length-scale 1.2

Output is 16 kHz mono wav (ready for wake word experiments / training).
Use --raw to keep the voice's native sample rate (22050 Hz).
"""
import argparse
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="text to speak")
    parser.add_argument(
        "--voice",
        default="denis",
        help="voice short name (denis, dmitri, irina, ruslan) or path to .onnx",
    )
    parser.add_argument("--out", default="output.wav", help="output wav path")
    parser.add_argument("--length-scale", type=float, default=1.0, help=">1 slower, <1 faster")
    parser.add_argument("--noise-scale", type=float, default=0.667, help="audio variability")
    parser.add_argument("--noise-w-scale", type=float, default=0.8, help="phoneme duration variability")
    parser.add_argument("--raw", action="store_true", help="keep native 22050 Hz instead of 16 kHz")
    args = parser.parse_args()

    voice_path = Path(args.voice)
    if not voice_path.suffix:
        voice_path = Path("voices") / f"ru_RU-{args.voice}-medium.onnx"
    if not voice_path.exists():
        print(f"Voice not found: {voice_path}")
        return 1

    voice = PiperVoice.load(str(voice_path))
    with wave.open(args.out, "wb") as wf:
        voice.synthesize_wav(
            args.text,
            wf,
            syn_config=SynthesisConfig(
                length_scale=args.length_scale,
                noise_scale=args.noise_scale,
                noise_w_scale=args.noise_w_scale,
            ),
        )

    if not args.raw:
        import librosa
        import scipy.io.wavfile

        sr, audio = scipy.io.wavfile.read(args.out)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        scipy.io.wavfile.write(args.out, 16000, (audio_16k * 32767).astype(np.int16))

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
