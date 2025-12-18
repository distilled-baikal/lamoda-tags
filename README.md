# Lamoda Tags

Автоматическая генерация и предсказание тегов для товаров Lamoda на основе отзывов покупателей.

## Описание

Проект включает два подхода к генерации тегов:
1. **LLM-генерация** — генерация тегов через языковую модель (GPT и др.)
2. **ML-модель** — обучение классификатора для предсказания тегов (RuBERT)

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
| `lamoda_reviews_sampled.csv` | Сгруппированные отзывы |
| `lamoda_reviews_sampled_with_tags.csv` | Отзывы + LLM теги |
| `tag_pulls_raw.csv` | Сырой пул тегов |
| `tag_pulls_best.csv` | Финальный пул тегов |
| `tag_mappings.csv` | Маппинг old → new |
| `lamoda_reviews_sampled_with_pull_tags.csv` | Финальные данные |

---

## Пример тегов

✅ Хорошие: `Мягкая пена`, `Приятный запах`, `Легкие`, `Заедает замок`

❌ Плохие: `Хороший товар`, `Рекомендую`, `В размер`, `Популярный`
