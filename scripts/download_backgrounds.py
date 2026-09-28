"""Downloads background audio sources (ESC-50 environmental sounds + FMA xsmall music),
converts everything to 16 kHz mono wav, and cleans up the originals.

Substitutes the notebook's AudioSet shard to save disk space while keeping
household/environmental noise diversity for augmentation.
"""
import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile  # noqa: F401  (datasets Audio backend)

FFMPEG = "ffmpeg"


def convert_to_16k(src: str, dst_dir: str) -> None:
    name = Path(src).stem + ".wav"
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        return
    subprocess.run(
        [
            FFMPEG,
            "-v", "error",
            "-y", "-i", src,
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            dst,
        ],
        check=True,
    )


def parallel_convert(files, dst_dir, workers=6):
    os.makedirs(dst_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda f: convert_to_16k(f, dst_dir), files))


def esc50():
    zip_path = "esc50.zip"
    if not os.path.isdir("esc50_16k"):
        if not os.path.exists(zip_path):
            print("Downloading ESC-50...", flush=True)
            subprocess.run(
                [
                    "curl", "-sL", "-o", zip_path,
                    "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
                ],
                check=True,
            )
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if "/audio/" in m and m.endswith(".wav")]
            zf.extractall("esc50_tmp", members=members)
        src_files = [str(p) for p in Path("esc50_tmp").glob("**/*.wav")]
        print(f"ESC-50: converting {len(src_files)} clips", flush=True)
        parallel_convert(src_files, "esc50_16k")
        subprocess.run(["rm", "-rf", "esc50_tmp", zip_path], check=True)
    print(f"ESC50_DONE {len(os.listdir('esc50_16k'))}", flush=True)


def fma():
    zip_path = "fma_xs.zip"
    if not os.path.isdir("fma_16k"):
        if not os.path.exists(zip_path):
            print("Downloading FMA xsmall...", flush=True)
            subprocess.run(
                [
                    "curl", "-sL", "-o", zip_path,
                    "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip",
                ],
                check=True,
            )
        os.makedirs("fma_tmp", exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall("fma_tmp")
        src_files = [str(p) for p in Path("fma_tmp").glob("**/*.mp3")]
        print(f"FMA: converting {len(src_files)} clips", flush=True)
        parallel_convert(src_files, "fma_16k")
        subprocess.run(["rm", "-rf", "fma_tmp", zip_path], check=True)
    print(f"FMA_DONE {len(os.listdir('fma_16k'))}", flush=True)


if __name__ == "__main__":
    esc50()
    fma()
    print("BACKGROUNDS_DONE")
