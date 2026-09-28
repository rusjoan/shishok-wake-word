#!/bin/bash
# Generates wake word positives ("шишок") and adversarial near-word negatives
# with 4 Russian Piper voices, varying length and noise scales.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=piper-sample-generator
PY=./.venv/bin/python

$PY -m piper_sample_generator positives_text.txt \
  --max-samples 6000 \
  --model voices/ru_RU-denis-medium.onnx \
  --model voices/ru_RU-dmitri-medium.onnx \
  --model voices/ru_RU-irina-medium.onnx \
  --model voices/ru_RU-ruslan-medium.onnx \
  --length-scales 0.85 1.0 1.15 1.3 \
  --noise-scales 0.667 0.75 0.85 1.0 \
  --output-dir positives_raw 2>&1 | grep -vE "DEBUG|FutureWarning|warnings.warn" | tail -2

echo "POSITIVES_DONE"

$PY -m piper_sample_generator adversarial_text.txt \
  --max-samples 1300 \
  --model voices/ru_RU-denis-medium.onnx \
  --model voices/ru_RU-dmitri-medium.onnx \
  --model voices/ru_RU-irina-medium.onnx \
  --model voices/ru_RU-ruslan-medium.onnx \
  --length-scales 0.85 1.0 1.15 1.3 \
  --noise-scales 0.667 0.75 0.85 1.0 \
  --output-dir adversarial_raw 2>&1 | grep -vE "DEBUG|FutureWarning|warnings.warn" | tail -2

echo "ADVERSARIAL_DONE"
