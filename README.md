# Lamoda Tags

Автоматическая генерация и предсказание тегов для товаров Lamoda на основе отзывов покупателей.

## Описание

Проект включает два подхода к генерации тегов:
1. **LLM-генерация** — генерация тегов через языковую модель (GPT и др.)
2. **ML-модель** — обучение классификатора для предсказания тегов (RuBERT)

Дополнительно в репозитории есть **инференс через LoRA-адаптеры (PEFT)** для 5 укрупнённых типов товаров (Bags / Shoes / Clothes / Beauty_Accs / Accs+Home_Accs) поверх одного базового HF-моделя.

## Установка

```bash
# Установить зависимости
pip install uv
uv sync

# Для разработки (jupyter, визуализация)
uv sync --extra dev
```

## Конфигурация LLM API

Для скриптов генерации тегов через LLM:

```bash
export OPENAI_BASE_URL="https://your-api-endpoint"
export OPENAI_TOKEN="your-api-key"
export OPENAI_MODEL="model-name"
export OPENAI_EMBED_MODEL="text-embedding-qwen3-embedding-0.6b"
export MAX_CONCURRENT="2"
export SIMILARITY_THRESHOLD="0.91"
```

---

## Часть 1: Генерация тегов через LLM

### Скрипты

#### `scripts/sample.py`

**Предобработка и сэмплирование данных отзывов.**

- Читает исходный файл `lamoda_reviews.csv`
- Сэмплирует до 10,000 строк для каждого `good_type`
- Группирует отзывы по `product_sku`
- Фильтрует товары с < 4 отзывами

```bash
python scripts/sample.py
```

**Вход:** `lamoda_reviews.csv` → **Выход:** `lamoda_reviews_sampled.csv`

---

#### `scripts/gen_tags.py`

**Генерация тегов с помощью LLM.**

- Асинхронная обработка с настраиваемым параллелизмом
- Автоматическое разбиение на чанки
- Идемпотентность (продолжает с места остановки)

```bash
python scripts/gen_tags.py           # Полный запуск
python scripts/gen_tags.py --test    # Тестовый режим (5 товаров)
python scripts/gen_tags.py --reset-failed  # Перезапуск неудачных
```

**Вход:** `lamoda_reviews_sampled.csv` → **Выход:** `lamoda_reviews_sampled_with_tags.csv`

---

#### `scripts/gen_tag_pulls.py`

**Создание пула уникальных тегов по типам товаров.**

```bash
python scripts/gen_tag_pulls.py
```

**Вход:** `lamoda_reviews_sampled_with_tags.csv` → **Выход:** `tag_pulls_raw.csv`

---

#### `scripts/gen_best_tags.py`

**Консолидация тегов через эмбеддинги и LLM.**

- Кластеризация похожих тегов
- Объединение синонимов
- Создание маппинга old → new

```bash
python scripts/gen_best_tags.py
```

**Вход:** `tag_pulls_raw.csv`  
**Выход:** `tag_pulls_best.csv`, `tag_mappings.csv`

---

#### `scripts/update_reviews_with_pull_tags.py`

**Обновление тегов согласно пулу.**

```bash
python scripts/update_reviews_with_pull_tags.py
```

**Вход:** `lamoda_reviews_sampled_with_tags.csv`, `tag_mappings.csv`, `tag_pulls_best.csv`  
**Выход:** `lamoda_reviews_sampled_with_pull_tags.csv`

---

### Полный LLM-пайплайн

```bash
python scripts/sample.py
python scripts/gen_tags.py
python scripts/gen_tag_pulls.py
python scripts/gen_best_tags.py
python scripts/update_reviews_with_pull_tags.py
```

---

## Часть 2: ML-модель для предсказания тегов

После генерации тегов через LLM можно обучить ML-модель для быстрого предсказания.

### Датасет и метрики

- 2 654 товара в итоговом CSV (`lamoda_reviews_sampled_with_pull_tags.csv`), 8 `good_type`.
- Среднее число тегов на товар: 2.61 (медиана 2, p90 ≈ 5), около 8.4 % товаров остаются без тегов и отбрасываются после `apply_label_vocab`.
- 1 095 уникальных тегов с длинным хвостом: медианная частота 2, но топ‑50 покрывают ~51 % упоминаний.
- Из-за дисбаланса macro-F1 всегда будет низким; основной показатель — micro-F1.
- Категорийный приор (топ‑K самых частых тегов внутри `good_type`) уже даёт 0.31–0.36 micro-F1, поэтому модель обязана как минимум превзойти этот уровень.
- Модель дополнительно использует:
  - allow-list тегов по категории (`filter_probs_by_good_type`)
  - priors в логит-пространстве (сглаженные частоты по `good_type`)
  - тюнинг глобального/по-тегового порога по валидации без принудительного `top_k` (для честной метрики); `top_k` применяется только на инференсе.

### Архитектура

- **Модель**: RuBERT (`DeepPavlov/rubert-base-cased`)
- **Задача**: Multi-label classification
- **Loss**: Focal Loss (лучше работает с дисбалансом классов)
- **Особенности**:
  - Chunking длинных текстов с overlap
  - Фильтрация тегов по категории товара
  - Per-label threshold tuning

### Структура модуля `model/`

```
model/
├── __init__.py      # Экспорты
├── config.py        # Конфигурация и гиперпараметры
├── data.py          # Загрузка и препроцессинг данных
├── losses.py        # Focal Loss, Asymmetric Loss
├── train.py         # Скрипт обучения
└── predict.py       # Инференс
```

### Обучение модели

```bash
# Из корня проекта
python -m model.train

# Или через entry point
train-tags
```

**Конфигурация** в `model/config.py`:

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    num_train_epochs: int = 4
    loss_type: str = "focal"  # "bce", "focal", "asl"
    focal_gamma: float = 2.0
    ...
```

**Артефакты** сохраняются в:
- `model/output/best_model/` — веса модели
- `model/artifacts/` — пороги, метки, конфиг

### Инференс

```python
from model import TagPredictor

# Загрузка модели
predictor = TagPredictor.load(
    model_path="model/output/best_model",
    artifacts_path="model/artifacts"
)

# Предсказание
tags = predictor.predict(
    text="Полотенце | Home_Accs | BATH TOWELS | Отличные полотенца, мягкие и впитывают.",
    good_type="Home_Accs"
)

print(tags)
# [('мягкие', 0.92), ('хорошо впитывают', 0.87), ...]
```

#### LLM rerank (опционально)

Чтобы дешёво улучшить качество на небольшом числе кандидатов:
1. Модель даёт top‑N (например, 20) тегов по вероятности.
2. LLM проверяет каждый тег по эталонному тексту и оставляет только релевантные (JSON со score 0..1).

Используются те же `OPENAI_BASE_URL`, `OPENAI_TOKEN`, `OPENAI_MODEL`, что и для LLM-скриптов.  
Включение в CLI:

```bash
predict-tags \
  --text "Отличные кроссовки, легкие и удобные" \
  --good-type "Shoes" \
  --llm-rerank \
  --rerank-top-n 20 \
  --rerank-max-output 5
```

В коде можно передать `LLMReranker` в `TagPredictor.predict(..., llm_reranker=reranker)`.  
LLM видит только 20 коротких кандидатов, поэтому токенизация ответа минимальна.

### Streamlit UI

Простой интерфейс для просмотра товаров, отзывов и тегов модели (по умолчанию без «истинных» тегов в датасете):

```bash
streamlit run app.py
```

![Lamoda Tags Explorer](imgs/app.png)

Приложение:
- Загружает товары из `lamoda_reviews_sampled.csv` (без колонки `tags`).
- Позволяет выбрать SKU и посмотреть исходные отзывы.
- Показывает предсказания тегов с вероятностями.
- Умеет включать LLM-rerank (при наличии `OPENAI_*` переменных).
- В сайдбаре можно переключить режим:
  - **Use LoRA adapters (per-category)** (по умолчанию включено)
  - выключить и использовать baseline модель `TagPredictor` из `model/output/best_model` (если она есть локально)

#### LoRA-режим (рекомендуется)

Streamlit (и `model/LoraTagPredictor`) берёт:
- **LoRA-адаптер** по `good_type` из папки `lora_model/`
- **базовую модель** из локального `hf_models/` (если скачана), иначе попытается взять с HuggingFace Hub.

Маппинг по `good_type`:
- `Bags` → `lora_bert_output_bags`
- `Shoes` → `lora_bert_output_shoes`
- `Clothes` → `lora_bert_output_clothes`
- `Beauty_Accs` → `lora_bert_output_beauty`
- `Accs`, `Home_Accs`, и любые другие/неизвестные категории → `lora_bert_output_accs`

##### Скачать LoRA-адаптеры

```bash
python scripts/download_lora_models.py --out-dir lora_model
```

Ожидаемая структура:

```
lora_model/
  lora_bert_output_bags/
  lora_bert_output_accs/
  lora_bert_output_beauty/
  lora_bert_output_clothes/
  lora_bert_output_shoes/
```

##### Скачать базовую HF-модель локально (веса + токенайзер)

```bash
python scripts/download_hf_model.py --model-id deepvk/user-bge-m3 --local-dir hf_models/user-bge-m3
```

Важно: в `hf_models/<...>/` должны быть файлы весов (`*.safetensors` или `pytorch_model*.bin`).  
Если там только `tokenizer.json/config.json`, значит скачался не тот формат (например, sentence-transformers без весов для Transformers).

CLI:

```bash
predict-tags --text "Отличные кроссовки, легкие и удобные" --good-type "Shoes"
```

### Улучшения по сравнению с notebook

| Аспект | Было | Стало |
|--------|------|-------|
| Loss | BCE | Focal Loss (γ=2.0) |
| Scheduler | Нет | Cosine с warmup |
| Early stopping | Нет | patience=2 |
| Batch size | 8 | 8 × 2 (grad accum) |
| min_freq | 10 | 5 |
| Dropout | default | 0.1 |

---

## Структура данных

| Файл | Описание |
|------|----------|
| `lamoda_reviews.csv` | Исходные отзывы (не в репо) |
| `lamoda_reviews_sampled.csv` | Сгруппированные отзывы (без тегов; используется в Streamlit по умолчанию) |
| `lamoda_reviews_sampled_with_tags.csv` | Отзывы + LLM теги |
| `tag_pulls_raw.csv` | Сырой пул тегов |
| `tag_pulls_best.csv` | Финальный пул тегов |
| `tag_mappings.csv` | Маппинг old → new |
| `lamoda_reviews_sampled_with_pull_tags.csv` | Финальные данные |

---

## Пример тегов

✅ Хорошие: `Мягкая пена`, `Приятный запах`, `Легкие`, `Заедает замок`

❌ Плохие: `Хороший товар`, `Рекомендую`, `В размер`, `Популярный`
