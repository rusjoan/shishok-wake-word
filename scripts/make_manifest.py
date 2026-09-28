"""Writes the ESPHome model manifest for the trained "шишок" model.

The probability cutoff is chosen from the streaming quantized test results:
the smallest cutoff whose estimated false accepts per hour <= 0.5, so recall
is maximized subject to a low false accept rate.
"""
import json
import re
import sys

CUTOFF_FAPH_LIMIT = 0.5

# Parses tflite_streaming_roc.txt lines: "Cutoff 0.98: frr=0.0767; faph=0.187"
line_re = re.compile(
    r"Cutoff ([0-9.]+): frr=([0-9.]+); faph=([0-9.]+)"
)

roc_name = sys.argv[1] if len(sys.argv) > 1 else "docs/v2-streaming-roc.txt"
rows = []
for line in open(roc_name):
    for cutoff, frr, faph in line_re.findall(line):
        rows.append((float(cutoff), float(frr), float(faph)))

if not rows:
    print("No cutoff results found in", roc_name)
    sys.exit(1)

print(f"{'cutoff':>8} {'FA/hour':>10} {'FRR %':>10} {'recall %':>10}")
for cutoff, frr, faph in rows:
    print(f"{cutoff:8.2f} {faph:10.3f} {frr*100:10.2f} {(1-frr)*100:10.2f}")

viable = [r for r in rows if r[2] <= CUTOFF_FAPH_LIMIT]
best = max(viable or rows, key=lambda r: 1 - r[1])
print(f"\nChosen cutoff: {best[0]} (FA/h={best[2]}, FRR={best[1]*100:.2f}%)")

manifest = {
    "type": "micro",
    "wake_word": "shishok",
    "author": "rusjoan",
    "website": "https://github.com/rusjoan",
    "model": "shishok.tflite",
    "trained_languages": ["ru"],
    "version": 2,
    "micro": {
        "probability_cutoff": best[0],
        "feature_step_size": 10,
        "sliding_window_size": 5,
        "tensor_arena_size": 26080,
        "minimum_esphome_version": "2024.7.0",
    },
}

with open("shishok.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Wrote shishok.json")
