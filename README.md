# shishok — wake word «шишок» для microWakeWord / ESPHome

Готовая модель детекции wake word **«шишок»** ([ʂɨˈʂok]) для микроконтроллеров
ESP32 (компонент [`micro_wake_word`](https://esphome.io/components/micro_wake_word)
в ESPHome), обученная с нуля на русскоязычной синтетике фреймворком
[microWakeWord](https://github.com/kahrendt/microWakeWord).

| Файл | Что это |
|---|---|
| `model/shishok.tflite` | квантизованная streaming-модель, 62 КБ |
| `model/shishok.json` | манифест для ESPHome (cutoff 0.94, окно 5) |
| `docs/manual-guide.md` | пошаговый мануал: как повторить всё обучение руками |
| `scripts/` | весь пайплайн: генерация данных, обучение-тесты, live-детекция на маке |

## Результаты

Оценка на held-out тестовых сплитах (аугментированные спектрограммы,
streaming-инференс, cutoff 0.94, окно 5):

| Набор | Срабатывания | median max p |
|---|---|---|
| Позитивы «шишок» (600 клипов) | **190/200 = 95%** | 0.996 |
| Слова-близнецы: шишка, мешок, пушок… (390) | 8/200 = **4%** | 0.027 |
| Реальная русская речь, Common Voice ru (250) | 0/200 = **0%** | 0.004 |

ROC квантизованной модели на ambient-аудио (DiPCo/CHiME6, полный файл —
[docs/v2-streaming-roc.txt](docs/v2-streaming-roc.txt)):

| cutoff | ложных срабатываний в час | FRR |
|---|---|---|
| 0.97 | 0.00 | 11.0% |
| 0.84 | 0.19 | 5.5% |
| 0.79 | 0.75 | 4.3% |
| 0.67 | 0.94 | 3.0% |

Архитектура: `mixednet` (MixConv, 4 блока, 26k параметров), вход — 40
mel-фич с шагом 10 мс (препроцессор micro_speech), окно 1.5 с, инференс
каждые 30 мс.

## Как пользоваться моделью

### ESPHome

Скопируйте `shishok.tflite` и `shishok.json` в конфиг (или на HTTP-хостинг):

```yaml
micro_wake_word:
  model:
    - model: shishok.json
  on_wake_word_detected:
    - voice_assistant: ...
```

`probability_cutoff` в манифесте подбирается по таблице ROC: выше — меньше
ложных срабатываний, ниже — надёжнее детекция.

### Живая проверка на macOS (микрофон)

```bash
# окружение — см. docs/manual-guide.md, шаги 0–2
python scripts/live_detect.py shishok.tflite --cutoff 0.94
```

Скармливает микрофон тому же препроцессору, что и на устройстве, и запускает
tflite каждые 30 мс. При выходе (Ctrl+C) печатает максимум вероятности за
сессию — по нему удобно подбирать cutoff. `--save-audio session.wav` пишет
сырой фид (отладка + заготовка записей для дообучения).

### Оценка на своих клипах

```bash
python scripts/smoke_test.py shishok.tflite
```

Прогоняет модель по папкам с wav (позитивы, слова-близнецы, речь, шумы) и
показывает долю срабатываний и медианную вероятность.

## Обучение своей модели (или дообучение этой)

Полная инструкция — [docs/manual-guide.md](docs/manual-guide.md). Коротко:

1. Клонируются `microWakeWord` и `piper-sample-generator`, ставится
   окружение (+ патч для macOS), скачиваются 4 русских голоса Piper.
2. Генерируется ~6000 позитивов «шишок» и ~3900 **адверсариальных негативов**
   (шишка, мешок, пушок… — критично против ложных срабатываний).
3. Реальная русская речь берётся стримом из Common Voice ru; ambient-шумы —
   готовые наборы microWakeWord; реверберация — MIT RIR; фоны — ESC-50 + FMA.
4. Спектрограммы → mmap, обучение 20k шагов (2 фазы), квантизация → tflite.
5. Тест: ROC на ambient + FPR на адверсариях (второе — самая частая проблема).

Ключевой опыт: **бенчмарк по ambient-аудио не ловит ложные срабатывания на
близких словах**. Наша первая версия показывала отличные официальные метрики
(0.375 FA/h, FRR 5.7%), но срабатывала на 19.5% адверсариальных клипов.
Усиление адверсариев (больше слов-близнецов, penalty ×2, SpecAugment) снизило
FPR до 4%.

## Ожидаемая структура каталога

```text
.
├── shishok.tflite, shishok.json     # модель
├── training_parameters_v2.yaml      # конфиг обучения
├── positives_text.txt               # фразы с «шишок» для синтеза
├── adversarial_text.txt             # слова-близнецы
├── scripts/                         # пайплайн
├── docs/                            # мануал, ROC, pip freeze
├── patches/                         # macOS-патч для microWakeWord
├── microWakeWord/                   # git clone (не в репо)
├── piper-sample-generator/          # git clone (не в репо)
├── voices/                          # Piper-голоса (скачиваются)
└── ...                              # данные, фичи, логи (генерируются)
```

## Данные и кредиты

- [microWakeWord](https://github.com/kahrendt/microWakeWord) — фреймворк
  обучения (Apache-2.0) и pre-generated негативные наборы
  ([kahrendt/microwakeword](https://huggingface.co/datasets/kahrendt/microwakeword), CC-BY-NC-4.0);
- [Piper](https://github.com/rhasspy/piper) и
  [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) —
  синтез речи (голоса ru_RU: denis, dmitri, irina, ruslan);
- [Common Voice 17 ru](https://huggingface.co/datasets/fsicoli/common_voice_17_0) (CC-0) —
  реальная русская речь в негативах;
- [ESC-50](https://github.com/karolpiczak/ESC-50) (CC-BY-NC),
  [FMA xsmall](https://huggingface.co/datasets/mchl914/fma_xsmall) — фоновые шумы;
- [MIT Environmental IRs](https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses) —
  реверберация.

## Лицензии

- **Код репозитория** — MIT (см. [LICENSE](LICENSE)).
- **Модель `shishok.tflite`** — обучение использовало pre-generated негативы
  (CC-BY-NC-4.0) и ESC-50 (CC-BY-NC), поэтому модель распространяется на
  условиях **CC-BY-NC-4.0**: некоммерческое личное использование. Это же
  ограничение накладывает оригинальный ноутбук microWakeWord на любые модели,
  обученные с этими данными.

## Ограничения

- Модель обучена на TTS-синтетике: на незнакомых голосах вероятности ниже
  тестовых. Главный апгрейд — 50–200 реальных записей слова (можно нарезать
  из `--save-audio` live-детектора) и дообучение.
- FA/h оценивался на английском ambient (DiPCo); для русской среды стоит
  собрать свою ambient-запись и прогнать `scripts/smoke_test.py`.
