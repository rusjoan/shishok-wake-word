"""Streams the first N clips of the Common Voice 17.0 (ru) test shard.

The shard tar is ~380 MB; we stop the download once enough clips are extracted
so we never store the whole archive on disk.
"""
import os
import signal
import subprocess
import sys
import tarfile

URL = "https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main/audio/ru/test/ru_test_0.tar"
TARGET_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
OUT_DIR = "cv_ru"

os.makedirs(OUT_DIR, exist_ok=True)
existing = len([f for f in os.listdir(OUT_DIR) if f.endswith(".mp3")])
if existing >= TARGET_CLIPS:
    print(f"CV_RU_DONE already {existing}")
    sys.exit(0)

proc = subprocess.Popen(["curl", "-sL", URL], stdout=subprocess.PIPE, bufsize=10 * 1024 * 1024)
count = existing

try:
    with tarfile.open(fileobj=proc.stdout, mode="r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if not name.endswith(".mp3"):
                continue
            dst = os.path.join(OUT_DIR, name)
            if os.path.exists(dst):
                continue
            with open(dst, "wb") as f:
                f.write(tf.extractfile(member).read())
            count += 1
            if count % 250 == 0:
                print(f"extracted {count}", flush=True)
            if count >= TARGET_CLIPS:
                break
except (EOFError, OSError) as e:
    print(f"stream ended: {e}", flush=True)

proc.send_signal(signal.SIGTERM)
proc.wait()
print(f"CV_RU_DONE {count}")
