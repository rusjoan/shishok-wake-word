# Пошаговый мануал: обучаем wake word «шишок» на microWakeWord вручную

Для middle-специалиста. Всё делается на macOS (Apple Silicon), но почти всё
совпадает и для Linux. Время: ~3–4 часа, из которых обучение ~40 минут.
Свободное место: ~15 ГБ.

Результат: квантизованная streaming-модель `.tflite` (~62 КБ) + манифест для
ESPHome + проверенные метрики.

---

## Шаг 0. Требования и системные зависимости

```bash
brew install python@3.10 ffmpeg git unzip
python3.10 --version   # нужно именно 3.10 (README microWakeWord рекомендует его)
ffmpeg -version
```

Проверьте место: `df -h /` — нужно ≥15 ГБ свободно.

> microWakeWord тренирует модель через TensorFlow на CPU. GPU (MPS) не
> используется — код сам его отключает на ARM Mac, это нормально.

## Шаг 1. Репозитории

```bash
mkdir shishok && cd shishok
git clone --depth 1 https://github.com/kahrendt/microWakeWord.git
git clone --depth 1 https://github.com/rhasspy/piper-sample-generator.git
```

## Шаг 2. Виртуальное окружение и зависимости

```bash
python3.10 -m venv .venv
./.venv/bin/pip install --upgrade pip

# Форки (обязательно!):
# - pymicro-features от puddly — собирается на macOS (у оригинала проблема с версией C++)
./.venv/bin/pip install 'git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version'
# - audio-metadata без пина attrs (ломает импорты)
./.venv/bin/pip install 'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'

# Сам фреймворк (тянет tensorflow 2.x, numpy 2.x, audiomentations, mmap_ninja и т.д.)
./.venv/bin/pip install -e ./microWakeWord

# Патч совместимости macOS (форк pymicro-features использует CamelCase API):
git apply patches/microwakeword-macos-pymicro-features.patch

# Генератор сэмплов (тянет piper-tts, torch, onnxruntime)
./.venv/bin/pip install ./piper-sample-generator

# Фиксы совместимости:
./.venv/bin/pip install "datasets==3.6.0"   # 5.x требует torchcodec для аудио — не нужен
./.venv/bin/pip install "setuptools<81"      # webrtcvad импортирует pkg_resources, удалённый в 81
./.venv/bin/pip install audiomentations==0.43.1  # пин 0.33.0 от piper-sample-generator без AddColorNoise
./.venv/bin/pip install tensorboard sounddevice librosa
```

Проверка:

```bash
./.venv/bin/python -c "import tensorflow as tf; print(tf.__version__)"   # 2.18+
PYTHONPATH=piper-sample-generator ./.venv/bin/python -m piper_sample_generator --help
```

## Шаг 3. Русские голоса Piper

В репо `rhasspy/piper-voices` есть 4 русских medium-голоса: denis, dmitri,
irina, ruslan (мужские denis/dmitri/ruslan, женский irina).

```bash
mkdir voices
for v in denis dmitri irina ruslan; do
  curl -L -o "voices/ru_RU-$v-medium.onnx" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/$v/medium/ru_RU-$v-medium.onnx"
  curl -L -o "voices/ru_RU-$v-medium.onnx.json" \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/$v/medium/ru_RU-$v-medium.onnx.json"
done
```

Проверка синтеза (должен появиться валидный wav ~0.6 с):

```bash
PYTHONPATH=piper-sample-generator ./.venv/bin/python -m piper_sample_generator "Шишок." \
  --max-samples 3 --model voices/ru_RU-denis-medium.onnx --output-dir test_out
ffprobe test_out/0.wav
```

## Шаг 4. Аудио для аугментаций (RIR + фоны)

### 4.1 Room impulse responses (MIT)

```bash
./.venv/bin/python - <<'EOF'
import datasets, scipy.io.wavfile, os, numpy as np
os.makedirs("mit_rirs", exist_ok=True)
ds = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses",
                           split="train", streaming=True)
for row in ds:
    name = row["audio"]["path"].split("/")[-1]
    scipy.io.wavfile.write(os.path.join("mit_rirs", name), 16000,
                           (row["audio"]["array"]*32767).astype(np.int16))
EOF
ls mit_rirs | wc -l   # ~170+
```

### 4.2 Фоновые шумы: ESC-50 (быт/город) и FMA xsmall (музыка)

```bash
# ESC-50
curl -L -o esc50.zip https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
unzip -q esc50.zip -d esc50_tmp
mkdir -p esc50_16k
for f in esc50_tmp/*/audio/*.wav; do
  ffmpeg -v error -y -i "$f" -ac 1 -ar 16000 -sample_fmt s16 "esc50_16k/$(basename $f)"
done
rm -rf esc50_tmp esc50.zip   # экономим место

# FMA xsmall
curl -L -o fma_xs.zip "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"
unzip -q fma_xs.zip -d fma_tmp
mkdir -p fma_16k
for f in $(find fma_tmp -name '*.mp3'); do
  ffmpeg -v error -y -i "$f" -ac 1 -ar 16000 -sample_fmt s16 "fma_16k/$(basename ${f%.mp3}).wav"
done
rm -rf fma_tmp fma_xs.zip
```

> В оригинальном ноутбуке используется AudioSet — но один шарда ~2.2 ГБ.
> ESC-50 + FMA дешевле по диску и хорошо покрывают домашние шумы и музыку.

## Шаг 5. Pre-generated негативные наборы (Hugging Face)

Эти наборы — спектрограммы в формате mmap, готовые для обучения (из
[kahrendt/microwakeword](https://huggingface.co/datasets/kahrendt/microwakeword)).

```bash
mkdir negative_datasets && cd negative_datasets
for f in dinner_party_eval.zip dinner_party.zip no_speech.zip; do
  curl -L -o "$f" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/$f"
  unzip -q "$f" -d . && rm "$f"
done
cd ..
```

Размеры: dinner_party_eval ~0.4 ГБ, dinner_party ~2.2 ГБ, no_speech ~8.9 ГБ
(внутри uint16-спектрограммы — не сжимаются в zip).

> Набор `speech.zip` (3 ГБ → ~12 ГБ) — английская начитка LibriSpeech. Мы его
> пропустили: диска не хватило, а речевые негативы закрыли dinner_party +
> Common Voice ru (шаг 6). Если места достаточно — скачайте и добавьте в конфиг
> с sampling_weight 10.

Структура каждого набора: `training/`, `validation/`, `testing/`,
`testing_ambient/` (не у всех) — внутри папки `*_mmap` с data.ninja и т.д.

## Шаг 6. Реальная русская речь (Common Voice ru)

Real-речь нужна, чтобы модель не срабатывала на обычную русскую речь. Полный
шард — 378 МБ, качаем стримом и останавливаемся на 2500 клипах:

```bash
./.venv/bin/python - <<'EOF'
import os, subprocess, signal, tarfile
URL = "https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main/audio/ru/test/ru_test_0.tar"
TARGET, count = 2500, 0
os.makedirs("cv_ru", exist_ok=True)
p = subprocess.Popen(["curl", "-sL", URL], stdout=subprocess.PIPE, bufsize=10*1024*1024)
try:
    with tarfile.open(fileobj=p.stdout, mode="r|*") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".mp3"):
                open(os.path.join("cv_ru", os.path.basename(m.name)), "wb").write(tf.extractfile(m).read())
                count += 1
                if count >= TARGET: break
except Exception as e: print("stream end:", e)
p.send_signal(signal.SIGTERM); p.wait()
print("done:", count)
EOF

# Конвертация в 16 кГц wav (быстрее грузится на этапе фич)
mkdir -p cv_ru_wav
for f in cv_ru/*.mp3; do
  ffmpeg -v error -y -i "$f" -ac 1 -ar 16000 -sample_fmt s16 "cv_ru_wav/$(basename ${f%.mp3}).wav"
done
```

## Шаг 7. Синтез сэмплов

### 7.1 Тексты позитивов

Файл `positives_text.txt` — фразы, слово всегда в конце (при обучении окно
1.5 с берётся с конца клипа). Пример:

```
Шишок.
Шишок!
шишок
А шишок!
Ну, шишок.
Скажи шишок.
Включи шишок.
...
```

### 7.2 Адверсариальные негативы (слова-близнецы)

Файл `adversarial_text.txt` — похожие по звучанию слова. Это КЛЮЧЕВОЙ набор
против ложных срабатываний на «шишка», «мешок», «пушок» и т.п.:

```
Шишка.
Шишки.
Мешок.
Пушок.
Петушок.
Мишка.
Окошки.
...
```

### 7.3 Генерация

```bash
PYTHONPATH=piper-sample-generator ./.venv/bin/python -m piper_sample_generator \
  positives_text.txt \
  --max-samples 6000 \
  --model voices/ru_RU-denis-medium.onnx \
  --model voices/ru_RU-dmitri-medium.onnx \
  --model voices/ru_RU-irina-medium.onnx \
  --model voices/ru_RU-ruslan-medium.onnx \
  --length-scales 0.85 1.0 1.15 1.3 \
  --noise-scales 0.667 0.75 0.85 1.0 \
  --output-dir positives_raw

# то же для adversarial_text.txt --max-samples 3900 --output-dir adversarial_raw
```

Что делают параметры:
- `--max-samples` — количество wav; генератор циклически перебирает пары
  (голос × length_scale × noise_scale) и тексты, каждый сэмпл уникален из-за
  сэмплирования шума в VITS.
- `--length-scales` — темп речи (0.85 быстрее, 1.3 медленнее).
- `--noise-scales` — вариативность тембра/интонации (0.667 — дефолт Piper).

Прослушайте 5–10 случайных файлов из `positives_raw` — все должны произносить
именно «шишок».

## Шаг 8. Генерация фич (спектрограммы → mmap)

Каждый клип аугментируется (шум, реверберация, EQ, gain) и превращается в
спектрограмму micro_speech preprocessor: 40 mel-фич, шаг 10 мс. Обучающие
спектрограммы дублируются со сдвигом на 1 кадр (`slide_frames=10`) — так
тренируем стриминговое поведение нестриминговой моделью.

Структура на выходе — по папке на сплит:

```
generated_augmented_features/
  training/wakeword_mmap     # 80% клипов × repeat 2 × slide 10
  validation/wakeword_mmap   # 10% × slide 10
  testing/wakeword_mmap      # 10% × slide 1 (настоящий streaming-тест)
```

Сплит детерминированный: `Clips(random_split_seed=10, split_count=0.1)` →
80/10/10.

Скрипт: `scripts/gen_features.py` (генерирует все три набора: позитивы,
cv_ru, адверсарии). Ключевые грабли:
- `RaggedMmap.from_generator` не создаёт родительские папки — создайте
  `.../training/` заранее;
- если генерация упала посреди — папка `wakeword_mmap` останется пустой и при
  перезапуске её надо удалить (проверяйте `len(RaggedMmap(dir))`);
- на macOS pymicro-features-форк имеет CamelCase API — нужен шим
  `_MicroFrontendCompat`; он уже в `patches/microwakeword-macos-pymicro-features.patch`.

## Шаг 9. Конфиг обучения

Файл `training_parameters_v2.yaml`. Разбор полей:

```yaml
features:               # списки наборов и их веса
  - features_dir: ...   # путь к папке со training/validation/testing
    sampling_weight: N  # вероятность попасть в батч (веса нормируются по сумме)
    penalty_weight: N   # множитель лосса за ошибку на этом наборе
    truth: true/false   # позитивы или негативы
    truncation_strategy: ...
        # truncate_start — окно 1.5 с берётся с конца (слово в конце клипа)
        # random       — случайное окно (длинные негативы)
        # split        — нарезка длинного ambient по 100 мс (для FA/h)
training_steps: [15000, 5000]         # фазы обучения, суммы шагов
learning_rates: [0.001, 0.0002]       # LR на каждую фазу
negative_class_weight: [20, 20]       # штраф за FP (больше — меньше ложных)
batch_size: 128
clip_duration_ms: 1500                # длина окна модели
eval_step_interval: 500               # валидация каждые N шагов
maximization_metric: "average_viable_recall"
minimization_metric: null             # НЕ пишите None! В YAML это строка "None"
```

## Шаг 10. Обучение

```bash
./.venv/bin/python -m microwakeword.model_train_eval \
  --training_config='training_parameters_v2.yaml' \
  --train 1 --restore_checkpoint 1 \
  --test_tf_nonstreaming 0 --test_tflite_nonstreaming 0 \
  --test_tflite_nonstreaming_quantized 0 --test_tflite_streaming 0 \
  --test_tflite_streaming_quantized 1 \
  --use_weights "best_weights" \
  mixednet \
  --pointwise_filters "64,64,64,64" \
  --repeat_in_block "1, 1, 1, 1" \
  --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
  --residual_connection "0,0,0,0" \
  --first_conv_filters 32 --first_conv_kernel_size 5 --stride 3
```

На M1 Pro — ~20 шагов/с, 20 000 шагов ≈ 35–40 мин. Упало/прервали — просто
перезапустите той же командой (`--restore_checkpoint 1` продолжит).

Наблюдение:

```bash
./.venv/bin/python -m tensorboard.main --logdir trained_models/wakeword_v2/logs
```

## Шаг 11. Чтение результата

В логе после обучения — тест квантизованной streaming-модели, таблица вида:

```
Cutoff 0.98: frr=0.0767; faph=0.187
Cutoff 0.94: frr=0.0567; faph=0.375
```

- `frr` — false reject rate: доля пропущенных «шишок» (меньше — лучше).
- `faph` — оценочные ложные срабатывания в час на ambient-аудио (меньше — лучше).

Выбирайте cutoff под задачу: для голосового ассистента обычно 0.3–1.0 FA/h.

> ВАЖНО: faph считается на английском dinner-party аудио. Обязательно
> дополнительно прогоните `scripts/smoke_test.py` — он проверит FPR на
> адверсариальных словах и русской речи (это самая частая проблема).

## Шаг 12. Манифест для ESPHome

`shishok.json`:

```json
{
  "type": "micro",
  "wake_word": "shishok",
  "author": "you",
  "model": "shishok.tflite",
  "trained_languages": ["ru"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.94,
    "feature_step_size": 10,
    "sliding_window_size": 5,
    "tensor_arena_size": 26080,
    "minimum_esphome_version": "2024.7.0"
  }
}
```

`sliding_window_size: 5` — сколько инференсов подряд (по 30 мс) вероятность
должна держаться выше cutoff для срабатывания. Больше — надёжнее, но медленнее.

В ESPHome (`micro_wake_word` component): положите `shishok.tflite` и
`shishok.json` рядом с конфигом:

```yaml
micro_wake_word:
  model:
    - model: shishok.json
  on_wake_word_detected:
    - voice_assistant: ...
```

## Шаг 13. Живая проверка на макбуке

```bash
./.venv/bin/python scripts/live_detect.py \
  trained_models/wakeword_v2/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --cutoff 0.94
```

При первом запуске macOS спросит доступ к микрофону — разрешите. Скрипт гоняет
тот же пайплайн, что и на устройстве: mic 16 кГц → microfrontend (состояние
сохраняется между чанками) → tflite каждые 30 мс. Срабатывание печатается с
вероятностью и звуком `\a`.

Полезные флаги: `--cutoff`, `--window 5`, `--device` (индекс/имя из
`--list-devices`), `--verbose` (каждый инференс), `--save-audio session.wav`
(запись всего микрофона в wav — и для отладки, и как заготовка реальных
сэмплов для дообучения).

При выходе (Ctrl+C) скрипт печатает максимум вероятности за сессию — по нему
подбирайте `--cutoff` (обычно max_p − 0.05…0.1).

> Грабли: microfrontend-форк обрабатывает ≤10 мс аудио за вызов
> `ProcessSamples`, поэтому микрофонный поток надо скармливать чанками по
> 160 сэмплов (10 мс) с буфером остатка, а модель вызывать ровно раз на 3
> новых кадра — иначе часть речи теряется и слово «режется». Всё это уже
> учтено в `scripts/live_detect.py`.

Проверка отдельных фраз:

```bash
./.venv/bin/python scripts/say.py "Шишок." --voice irina --out test.wav
```

## Шаг 14. Как улучшать модель (типовой цикл)

1. **Реальные записи** — самый большой прирост. Запишите 50–200 файлов:
   свой «шишок» (разные дистанции, интонации) в `positives_raw/` и «шишка»,
   «мешок», случайную речь в `adversarial_raw/`/`cv_ru_wav/`. Перегенерируйте
   фичи и дообучите.
2. **Больше адверсариев** с высоким штрафом: `penalty_weight: 2.0`,
   `sampling_weight: 6.0` (в v2 это снизило FPR на близких словах).
3. **SpecAugment** в первой фазе: `time_mask_count: [2, 0]`,
   `freq_mask_count: [2, 0]`.
4. Если модель пропускает слово — поднимите `sampling_weight` позитивов или
   добавьте сэмплов; если ложные срабатывания — увеличивайте
   `negative_class_weight` (20 → 30) и адверсариев.
5. Обучайте в новую папку (`train_dir`), сравнивайте на одинаковых тестах.

## Troubleshooting

| Симптом | Причина | Лечение |
|---|---|---|
| `No module named 'pkg_resources'` | setuptools ≥ 81 | `pip install "setuptools<81"` |
| `To support encoding audio... 'torchcodec'` | datasets 5.x | `pip install "datasets==3.6.0"` |
| `'MicroFrontend' object has no attribute 'process_samples'` | macOS-форк pymicro-features с CamelCase API | примените `patches/microwakeword-macos-pymicro-features.patch` |
| `KeyError: 'None'` при обучении | в YAML `minimization_metric: None` — это строка | пишите `null` |
| `audiomentations has no attribute 'AddColorNoise'` | пин 0.33.0 от piper-sample-generator | `pip install audiomentations==0.43.1` |
| `FileNotFoundError ... wakeword_mmap` | mmap_ninja не создаёт родителей | `os.makedirs(..., exist_ok=True)` |
| Модель не грузится в ESPHome | малый tensor arena | поднять `tensor_arena_size` в манифесте |
| Много ложных срабатываний на «шишка/мешок» | мало адверсариев/вес | см. Шаг 14.2 |
| Модель не срабатывает на живой голос | синтетика ≠ реальный голос; см. также строку выше | посмотреть «Максимум вероятности за сессию» и снизить `--cutoff`; записать `--save-audio` и прослушать, тот ли устройство; добавить реальные записи в обучение (Шаг 14.1) |
| Диск кончился | негативные наборы большие | скачать только нужные zip, удалять архивы после распаковки |
