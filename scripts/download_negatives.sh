#!/bin/bash
# Downloads pre-generated negative spectrogram features (kahrendt/microwakeword on HF)
# and extracts them. Zip files are removed right after extraction to save disk.
set -e
mkdir -p negative_datasets
cd negative_datasets

for fname in dinner_party_eval.zip dinner_party.zip no_speech.zip speech.zip; do
    dir_name="${fname%.zip}"
    if [ ! -d "$dir_name" ]; then
        echo "=== Downloading $fname"
        curl -sL -o "$fname" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/$fname"
        echo "=== Extracting $fname"
        unzip -q "$fname" -d .
        rm -f "$fname"
        echo "=== Done $fname"
        df -h / | tail -1
    fi
done
echo "NEGATIVES_DONE"
