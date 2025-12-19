#!/usr/bin/env python3
"""
LoRA (PEFT) inference for per-category multi-label tag prediction.

This uses a single base model (`deepvk/user-bge-m3`) and loads one of 5
category-specific LoRA adapters from `lora_model/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "peft is required for LoRA inference. Install it with: uv add peft (or pip install peft)."
    ) from exc


# Default HF repo id; can be overridden with a local path (e.g. hf_models/user-bge-m3)
BASE_MODEL_NAME = "deepvk/user-bge-m3"


def _has_hf_weights(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    # Common HuggingFace weight filenames
    if (model_dir / "pytorch_model.bin").exists():
        return True
    if (model_dir / "model.safetensors").exists():
        return True
    if any(model_dir.glob("*.safetensors")):
        return True
    # Sharded weights
    if any(model_dir.glob("pytorch_model-*.bin")):
        return True
    if any(model_dir.glob("model-*.safetensors")):
        return True
    return False


bag_tags = [
    "Удобная",
    "Вместительная",
    "Компактная",
    "Много отделений",
    "Заедает замок",
    "Быстро изнашивается",
    "Долговечная",
    "Дешево выглядит",
    "Высокое качество",
    "Надежный замок",
    "Легко чистить",
    "Хорошая упаковка",
    "Есть дефекты",
    "Лёгкая",
    "Тяжёлая",
    "Прочная ткань",
    "Водонепроницаемая",
    "Неприятный запах",
    "Соответствует фото",
    "Хорошая цена",
    "Не соответствует фото",
    "Функциональная",
    "Универсальная",
]

cloth_tags = [
    "Удобный",
    "Приятный материал",
    "Сидит идеально",
    "Идет в размер",
    "Маломерит",
    "Большемерит",
    "Не жмет",
    "Дышит",
    "Качественный принт",
    "Для высокого роста",
    "Плохо пропускает воздух",
    "Просвечивает",
    "Не просвечивает",
    "Не садится после стирки",
    "Садится после стирки",
    "Скатывается ткань",
    "Ткань не скатывается",
    "Качественный",
    "Есть дефекты",
    "Цвет как на фото",
    "Цвет отличается",
]

shoes_tags = [
    "Удобные",
    "Комфортные",
    "Приятная кожа",
    "Приятная замша",
    "Не натирают",
    "Не давят",
    "На среднюю стопу",
    "Маломерят",
    "Большемерки",
    "Требуют разнашивания",
    "Амортизируют",
    "Теплые",
    "Лёгкие",
    "Есть дефекты",
    "Долго носятся",
    "Прочные",
    "Плохо держат ногу",
    "Идут в размер",
    "Не соответствует размеру",
    "Жмут в носке",
    "На широкую стопу",
    "На узкую стопу",
    "Промокают",
    "Не промокают",
    "Недостаточно тёплые",
    "Амортизирующие",
    "Легко чистятся",
    "Сильно натирают",
]

beauty_accs = [
    "Хорошо пенится",
    "Деликатно очищает",
    "Удобный",
    "Отличный состав",
    "Для ночного ухода",
    "Высокая плотность",
    "Питает кожу",
    "Удлиняет ресницы",
    "Сужает поры",
    "Быстро расходуется",
    "Нежный",
    "Легкий аромат",
    "Не оставляет пленки",
    "Освежает лицо",
    "Не стягивает кожу",
    "Слабое отшелушивание",
    "Совместим с кремами",
    "Для сухих волос",
    "Выраженный эффект",
    "Для чувствительной кожи",
    "Не сушит",
    "Освежает дыхание",
    "Отличное качество",
    "Подсушивает",
    "Мягко очищает",
    "Даёт объём",
    "Увлажняет надолго",
]

home_accs_plus_accs = [
    "Плотный",
    "Качественный",
    "Компактный",
    "Быстро расходуется",
    "Удобный",
    "Хорошая впитываемость",
    "Легко стирается",
    "Не деформируется",
    "Плохо просыхающий",
    "Есть дефекты",
    "Не ржавеет",
    "Прочные материалы",
    "Стойкий рисунок",
    "Блекнет после стирки",
    "Приятная ткань",
    "Не протекает",
    "Держит тепло",
    "Долговечный",
    "Дышит",
    "Надежный механизм",
    "Неудобный",
    "Не вызывает аллергии",
    "Маломерит",
    "Большемерит",
    "Образуют катышки",
]


GOOD_TYPE_TO_ADAPTER_KEY: Dict[str, str] = {
    "Bags": "bags",
    "Beauty_Accs": "beauty",
    "Clothes": "clothes",
    "Shoes": "shoes",
    # dataset has both; use same adapter
    "Accs": "accs",
    "Home_Accs": "accs",
}

ADAPTER_KEY_TO_DIRNAME: Dict[str, str] = {
    "bags": "lora_bert_output_bags",
    "accs": "lora_bert_output_accs",
    "beauty": "lora_bert_output_beauty",
    "clothes": "lora_bert_output_clothes",
    "shoes": "lora_bert_output_shoes",
}

ADAPTER_KEY_TO_TAGS: Dict[str, List[str]] = {
    "bags": bag_tags,
    "accs": home_accs_plus_accs,
    "beauty": beauty_accs,
    "clothes": cloth_tags,
    "shoes": shoes_tags,
}


@dataclass(frozen=True)
class _Bundle:
    name: str
    model: torch.nn.Module
    tokenizer: object
    tags: List[str]


class LoraTagPredictor:
    """
    Per-category LoRA adapter predictor.

    Returns the same output shape as TagPredictor.predict():
      List[(tag, score)] sorted by score desc.
    """

    def __init__(
        self,
        lora_root: str | Path = "lora_model",
        base_model_name: str | Path = BASE_MODEL_NAME,
        device: Optional[str] = None,
        threshold: float = 0.5,
        top_k: int = 8,
    ):
        self.lora_root = Path(lora_root)
        # Can be HF repo id (str) or a local directory (Path/str)
        self.base_model_name = str(base_model_name)
        self.threshold = float(threshold)
        self.top_k = int(top_k)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._bundles: Dict[str, _Bundle] = {}

    def _adapter_key(self, good_type: Optional[str]) -> str:
        gt = (good_type or "").strip()
        return GOOD_TYPE_TO_ADAPTER_KEY.get(gt, "accs")

    def _adapter_path(self, adapter_key: str) -> Path:
        dirname = ADAPTER_KEY_TO_DIRNAME[adapter_key]
        return self.lora_root / dirname

    def _load_bundle(self, adapter_key: str) -> _Bundle:
        if adapter_key in self._bundles:
            return self._bundles[adapter_key]

        adapter_path = self._adapter_path(adapter_key)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter folder not found: {adapter_path}. "
                f"Expected under {self.lora_root}/. "
                f"Run: python scripts/download_lora_models.py --out-dir {self.lora_root}"
            )

        tags = ADAPTER_KEY_TO_TAGS[adapter_key]

        base_model_src = self.base_model_name
        # If user passes a local dir, enforce that it contains actual model weights.
        try:
            p = Path(base_model_src)
            if p.exists():
                if not _has_hf_weights(p):
                    raise FileNotFoundError(
                        f"Base model folder exists but has no HF weights (*.safetensors / pytorch_model*.bin): {p}\n"
                        "You likely downloaded only a SentenceTransformers snapshot (tokenizer/config). "
                        "Re-download the full Transformers checkpoint with weights, e.g.:\n"
                        "  python scripts/download_hf_model.py --model-id deepvk/user-bge-m3 --local-dir hf_models/user-bge-m3"
                    )
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    str(p),
                    num_labels=len(tags),
                    local_files_only=True,
                )
            else:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_src,
                    num_labels=len(tags),
                )
        except OSError:
            # Fallback to default behavior for HF ids; keep error readable.
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_src,
                num_labels=len(tags),
            )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.to(self.device)
        model.eval()

        # Prefer tokenizer shipped with adapter if present; fall back to base model tokenizer.
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        except Exception:
            # If base_model_name is a local dir, keep it local-only.
            p = Path(self.base_model_name)
            if p.exists():
                tokenizer = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        bundle = _Bundle(name=adapter_key, model=model, tokenizer=tokenizer, tags=tags)
        self._bundles[adapter_key] = bundle
        return bundle

    @torch.no_grad()
    def predict(
        self,
        text: str,
        good_type: Optional[str] = None,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        llm_reranker=None,
        rerank_top_n: int = 20,
        rerank_max_output: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        adapter_key = self._adapter_key(good_type)
        bundle = self._load_bundle(adapter_key)

        inputs = bundle.tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = bundle.model(**inputs).logits[0]
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()

        thr = float(self.threshold if threshold is None else threshold)
        k = int(self.top_k if top_k is None else top_k)

        scored = [(bundle.tags[i], float(p)) for i, p in enumerate(probs)]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = [x for x in scored if x[1] >= thr]
        if not selected:
            selected = scored[:1]

        # Optional LLM rerank (same contract as TagPredictor)
        if llm_reranker is not None:
            top_n = max(1, min(int(rerank_top_n), len(scored)))
            candidates = scored[:top_n]
            try:
                reranked = llm_reranker.rerank(
                    text=text,
                    good_type=good_type,
                    candidates=candidates,
                    max_return=rerank_max_output or k,
                )
            except Exception:
                reranked = []
            if reranked:
                return reranked

        return selected[:k]


__all__ = ["LoraTagPredictor"]


