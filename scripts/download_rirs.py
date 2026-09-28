"""Downloads MIT environmental room impulse responses (16 kHz) for augmentation."""
import os

import datasets
import numpy as np
import scipy.io.wavfile

os.makedirs("mit_rirs", exist_ok=True)

rir_dataset = datasets.load_dataset(
    "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
)

count = 0
for row in rir_dataset:
    name = row["audio"]["path"].split("/")[-1]
    scipy.io.wavfile.write(
        os.path.join("mit_rirs", name),
        16000,
        (row["audio"]["array"] * 32767).astype(np.int16),
    )
    count += 1
    if count % 50 == 0:
        print(f"RIRs downloaded: {count}", flush=True)

print(f"RIRS_DONE {count}")
