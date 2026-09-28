"""Live wake word detection from the MacBook microphone.

Feeds the mic stream through the same micro_speech preprocessor used in
training (pymicro-features MicroFrontend keeps state across chunks, just like
on the device) and runs the quantized streaming tflite model every 30 ms.

Examples:
    ./.venv/bin/python scripts/live_detect.py trained_models/wakeword_v2/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
    ./.venv/bin/python scripts/live_detect.py path/to/model.tflite --cutoff 0.90 --window 5
    ./.venv/bin/python scripts/live_detect.py model.tflite --list-devices
    ./.venv/bin/python scripts/live_detect.py model.tflite --device 1
"""
import argparse
import collections
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from pymicro_features import MicroFrontend
from ai_edge_litert.interpreter import Interpreter

SAMPLE_RATE = 16000
FRAME_MS = 10          # spectrogram hop
STEP = 3               # frames per model inference (30 ms)
COOLDOWN_S = 2.0


def list_devices() -> None:
    print(sd.query_devices())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="path to the quantized streaming .tflite")
    parser.add_argument("--cutoff", type=float, default=0.94, help="probability threshold")
    parser.add_argument("--window", type=int, default=5, help="consecutive inferences above cutoff to trigger")
    parser.add_argument("--device", type=str, default=None, help="input device index or (partial) name")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_S, help="seconds before re-triggering")
    parser.add_argument("--save-audio", metavar="WAV", default=None,
                        help="also write the raw 16 kHz mic feed to this wav (debugging / collecting training samples)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print every inference")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return 0
    if not args.model:
        parser.error("model path is required (or use --list-devices)")

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)
        try:
            info = sd.query_devices(device)
        except (ValueError, sd.PortAudioError):
            print(f"Нет устройства с индексом {device} (см. --list-devices)")
            return 1
        if info["max_input_channels"] == 0:
            print(f"Устройство [{device}] {info['name']} не имеет входа (это выход)")
            return 1
    elif device is not None:
        candidates = [
            (i, d["name"])
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and device.lower() in d["name"].lower()
        ]
        if not candidates:
            print(f"No input device matching {device!r}")
            return 1
        device = candidates[0][0]

    interpreter = Interpreter(model_path=args.model, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale = out["quantization"][0]

    frontend = MicroFrontend()
    frame_buffer: collections.deque = collections.deque(maxlen=STEP)
    pending_frames = 0  # frames not yet consumed by the model
    prob_window: collections.deque = collections.deque(maxlen=args.window)
    triggered_at = 0.0

    detections = 0
    last_report = time.time()
    state = {"samples_in": 0, "max_p": 0.0, "max_abs": 0}
    byte_buffer = bytearray()
    CHUNK_BYTES = SAMPLE_RATE * 2 // 100  # 160 int16 samples = 10 ms
    wav_writer = None

    resolved = device if device is not None else sd.default.device[0]
    dev_info = sd.query_devices(resolved)
    dev_name = dev_info["name"]
    dev_rate = int(dev_info["default_samplerate"])
    # Open the stream at the device's native rate and decimate to 16 kHz —
    # asking the built-in mic for 16 kHz directly can yield silence on macOS.
    if dev_rate == SAMPLE_RATE:
        stream_rate, decim = SAMPLE_RATE, 1
    elif dev_rate % SAMPLE_RATE == 0:
        stream_rate, decim = dev_rate, dev_rate // SAMPLE_RATE
    else:
        stream_rate, decim = SAMPLE_RATE, 1
    print(f"Model: {args.model}")
    print(f"cutoff={args.cutoff}  window={args.window}")
    print(f"Вход: [{resolved}] {dev_name} @ {stream_rate} Hz"
          + (f" (→ {SAMPLE_RATE} Hz, /{decim})" if decim > 1 else ""))
    print("Говорите «шишок»… (Ctrl+C — выход)\n")

    def on_detect(p: float) -> None:
        nonlocal detections, triggered_at
        detections += 1
        triggered_at = time.time()
        print(
            f"\a>>> [{datetime.now().strftime('%H:%M:%S')}] WAKE WORD «ШИШОК» "
            f"обнаружен (p={p:.3f}) — срабатывание №{detections}"
        )

    def process_frames(new_frames) -> None:
        nonlocal pending_frames, triggered_at, last_report
        for frame in new_frames:
            frame_buffer.append(np.asarray(frame, dtype=np.float32))
            pending_frames += 1
            if pending_frames < STEP or len(frame_buffer) < STEP:
                continue
            # The streaming model must be invoked exactly once per STEP new
            # frames (every 30 ms) — its internal state advances on each call.
            pending_frames -= STEP
            window = np.concatenate(frame_buffer)  # (STEP*40,)
            q = np.round(window / in_scale + in_zp).astype(np.int8)
            interpreter.set_tensor(inp["index"], q.reshape(1, STEP, 40))
            interpreter.invoke()
            p = float(interpreter.get_tensor(out["index"])[0, 0]) * out_scale
            state["max_p"] = max(state["max_p"], p)

            prob_window.append(p)
            if args.verbose:
                bar = "#" * int(p * 40)
                print(f"\r p={p:.3f} {bar:<40}", end="")
            now = time.time()
            if now - last_report > 0.5 and not args.verbose:
                last_report = now
                bar = "#" * int(p * 40)
                print(f"\r p={p:.3f} {bar:<40}", end="")
            if (
                len(prob_window) == args.window
                and all(v >= args.cutoff for v in prob_window)
                and now - triggered_at > args.cooldown
            ):
                on_detect(p)
                prob_window.clear()

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"\n[audio status: {status}]", file=sys.stderr)
        pcm = indata[:, 0]
        state["max_abs"] = max(state["max_abs"], int(np.max(np.abs(pcm))))
        if decim > 1:
            # The stream runs at the device's native rate; bring it down to 16 kHz.
            pcm = resample_poly(pcm, 1, decim).astype(np.int16)
        data = pcm.tobytes()
        if wav_writer is not None:
            wav_writer.writeframes(data)
        state["samples_in"] += len(pcm)

        # The pymicro-features fork only consumes ~10 ms of audio per
        # ProcessSamples() call, so feed it strict 160-sample (10 ms) chunks.
        # Leftover bytes (if any) are carried over to the next callback.
        byte_buffer.extend(data)
        while len(byte_buffer) >= CHUNK_BYTES:
            chunk = bytes(byte_buffer[:CHUNK_BYTES])
            del byte_buffer[:CHUNK_BYTES]
            result = frontend.ProcessSamples(chunk)
            # features is a flat list: reshape to (num_frames, 40)
            new_frames = np.asarray(result.features, dtype=np.float32).reshape(-1, 40)
            process_frames(new_frames)

    if args.save_audio:
        import wave as wave_mod

        wav_writer = wave_mod.open(args.save_audio, "wb")
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(SAMPLE_RATE)
        print(f"Запись микрофона в {args.save_audio}")

    with sd.InputStream(
        samplerate=stream_rate,
        channels=1,
        dtype="int16",
        blocksize=int(stream_rate * FRAME_MS / 1000 * STEP),
        callback=audio_callback,
        device=device,
    ):
        start = time.time()
        warned_silent = False
        try:
            while True:
                time.sleep(0.1)
                if (
                    not warned_silent
                    and time.time() - start > 5.0
                    and state["samples_in"] > 0
                    and state["max_abs"] < 100
                ):
                    warned_silent = True
                    print(
                        "\n[!] Микрофон передаёт тишину. Возможные причины:\n"
                        "    - не тот вход открыт (сейчас: "
                        f"[{resolved}] {dev_name}). Запуститесь без --device\n"
                        "      (возьмётся системный вход) или по имени: "
                        "--device MacBook / --device rusjoan\n"
                        "    - в Системных настройках → Звук → Вход полоска не "
                        "двигается, когда вы говорите\n"
                        "    - Bluetooth-гарнитура перехватила системный вход\n"
                    )
        except KeyboardInterrupt:
            print(f"\nИтого срабатываний: {detections}")
            print(f"Максимум вероятности за сессию: {state['max_p']:.3f}")
            print(f"Максимум уровня входа (0–32767): {state['max_abs']}")
            if state["max_p"] < args.cutoff:
                print(
                    f"Порог {args.cutoff} не достигнут — попробуйте --cutoff "
                    f"{max(0.5, state['max_p'] - 0.05):.2f} или ниже"
                )
    if wav_writer is not None:
        wav_writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
