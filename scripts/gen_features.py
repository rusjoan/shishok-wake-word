"""Generates augmented spectrogram features (RaggedMmap) for:
  - positives ("шишок"),       -> generated_augmented_features/
  - Common Voice ru negatives  -> cv_ru_speech/
  - adversarial near-words     -> adversarial_features/

Mirrors basic_training_notebook.ipynb cells 5 and 7.
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mmap_ninja.ragged import RaggedMmap

from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration


def convert_cv_ru():
    """Converts Common Voice ru mp3 clips to 16 kHz mono wav for fast loading."""
    src_dir, dst_dir = "cv_ru", "cv_ru_wav"
    done = set(f.name for f in Path(dst_dir).glob("*.wav")) if os.path.isdir(dst_dir) else set()
    todo = [p for p in Path(src_dir).glob("*.mp3") if p.stem + ".wav" not in done]
    if not todo:
        print(f"cv_ru already converted ({len(done)} files)")
        return
    os.makedirs(dst_dir, exist_ok=True)

    def conv(p):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(p), "-ac", "1", "-ar", "16000",
             "-sample_fmt", "s16", str(Path(dst_dir) / (p.stem + ".wav"))],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(conv, todo))
    print(f"converted {len(todo)} cv_ru clips")


augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.1,
        "PitchShift": 0.1,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.75,
        "Gain": 1.0,
        "RIR": 0.5,
    },
    impulse_paths=["mit_rirs"],
    background_paths=["fma_16k", "esc50_16k"],
    background_min_snr_db=-5,
    background_max_snr_db=10,
    min_jitter_s=0.195,
    max_jitter_s=0.205,
)


def make_mmaps(clips, output_dir):
    """Generates training/validation/testing mmaps for a Clips object."""
    specs = {
        "training": dict(split="train", repeat=2, slide=10),
        "validation": dict(split="validation", repeat=1, slide=10),
        "testing": dict(split="test", repeat=1, slide=1),
    }
    for split, cfg in specs.items():
        out_dir = os.path.join(output_dir, split, "wakeword_mmap")
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        if os.path.exists(out_dir):
            try:
                n_existing = len(RaggedMmap(out_dir))
            except (IndexError, OSError):
                n_existing = 0
            if n_existing > 0:
                print(f"skip existing {out_dir} ({n_existing} samples)")
                continue
            subprocess.run(["rm", "-rf", out_dir], check=True)
        spectrograms = SpectrogramGeneration(
            clips=clips, augmenter=augmenter, slide_frames=cfg["slide"], step_ms=10
        )
        RaggedMmap.from_generator(
            out_dir=out_dir,
            sample_generator=spectrograms.spectrogram_generator(
                split=cfg["split"], repeat=cfg["repeat"]
            ),
            batch_size=100,
            verbose=True,
        )
        print(f"finished {out_dir}", flush=True)


def main():
    convert_cv_ru()

    print("=== positives", flush=True)
    positives = Clips(
        input_directory="positives_raw",
        file_pattern="*.wav",
        random_split_seed=10,
        split_count=0.1,
    )
    make_mmaps(positives, "generated_augmented_features")

    print("=== cv_ru negatives", flush=True)
    cv_ru = Clips(
        input_directory="cv_ru_wav",
        file_pattern="*.wav",
        random_split_seed=10,
        split_count=0.1,
    )
    make_mmaps(cv_ru, "cv_ru_speech")

    print("=== adversarial negatives", flush=True)
    adversarial = Clips(
        input_directory="adversarial_raw",
        file_pattern="*.wav",
        random_split_seed=10,
        split_count=0.1,
    )
    make_mmaps(adversarial, "adversarial_features")

    print("FEATURES_DONE")


if __name__ == "__main__":
    main()
