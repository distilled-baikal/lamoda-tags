#!/usr/bin/env python3
"""
Simple Streamlit UI for exploring Lamoda products and model tags.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from model.data import build_text, parse_pylist_or_json
from model.lora_predict import LoraTagPredictor
from model.predict import TagPredictor
from model.rerank import LLMReranker

DATA_PATH = Path("test_examples.csv")
MODEL_DIR = Path("model/output/best_model")
ARTIFACTS_DIR = Path("model/artifacts")
LORA_DIR = Path("lora_model")
HF_MODELS_DIR = Path("hf_models")


@st.cache_data(show_spinner=False)
def load_products(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    if "product_sku" not in df.columns or "comment_text" not in df.columns:
        raise ValueError("CSV must contain at least columns: product_sku, comment_text")

    df["comment_list"] = df["comment_text"].apply(parse_pylist_or_json)

    rows = []
    for sku, g in df.groupby("product_sku", sort=False):
        name = g["name"].iloc[0] if "name" in g.columns else ""
        good_type = g["good_type"].iloc[0] if "good_type" in g.columns else ""
        good_subtype = g["good_subtype"].iloc[0] if "good_subtype" in g.columns else ""

        comments = []
        for lst in g["comment_list"].tolist():
            comments.extend(lst)

        text = build_text(name, good_type, good_subtype, comments)
        rows.append(
            {
                "product_sku": sku,
                "name": name,
                "good_type": good_type,
                "good_subtype": good_subtype,
                "comment_list": comments,
                "text": text,
                "option_label": f"{sku} — {name or 'Без названия'}",
            }
        )

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_predictor(model_dir: Path, artifacts_dir: Path) -> TagPredictor:
    return TagPredictor.load(model_dir, artifacts_dir, device="cpu")


@st.cache_resource(show_spinner=False)
def load_lora_predictor(lora_dir: Path, hf_models_dir: Path) -> LoraTagPredictor:
    # Lazy-loads specific adapter on first use; keeps bundles cached in-memory.
    # Prefer local HF weights if present, to avoid any network calls.
    local_candidates = [
        hf_models_dir / "user-bge-m3",
        hf_models_dir / "USER-bge-m3",
        hf_models_dir / "deepvk_user-bge-m3",
        hf_models_dir / "deepvk_USER-bge-m3",
    ]
    base_model = next((p for p in local_candidates if p.exists()), "deepvk/user-bge-m3")
    return LoraTagPredictor(
        lora_root=lora_dir,
        base_model_name=base_model,
        device="cpu",
        threshold=0.5,
        top_k=8,
    )


@st.cache_resource(show_spinner=False)
def load_reranker() -> LLMReranker:
    return LLMReranker.from_env()


def main():
    st.set_page_config(page_title="Lamoda Tags Explorer", layout="wide")
    st.title("Lamoda Tags Explorer")
    st.caption("Выберите товар, просмотрите отзывы и посмотрите теги модели.")

    with st.sidebar:
        use_lora = st.checkbox(
            "Use LoRA adapters (per-category)",
            value=True,
            help="Загружает LoRA-адаптер по good_type из папки lora_model/ (Bags/Shoes/Clothes/Beauty_Accs/Accs+Home_Accs).",
        )
        use_llm = st.checkbox("LLM rerank (top-20)", value=False, help="Требуются OPENAI_* переменные")
        st.caption("LLM сверяет top-20 тегов с отзывами и оставляет только релевантные.")
        reranker: Optional[LLMReranker] = None
        if use_llm:
            try:
                reranker = load_reranker()
            except Exception as exc:
                st.error(f"LLM rerank недоступен: {exc}")
                reranker = None
        else:
            reranker = None

    if not DATA_PATH.exists():
        st.error(f"Не найден файл данных: {DATA_PATH}")
        return

    try:
        products = load_products(DATA_PATH)
    except Exception as exc:
        st.error(f"Не удалось загрузить данные: {exc}")
        return

    if products.empty:
        st.warning("Данные пусты.")
        return

    options = products["option_label"].tolist()
    default_index = 0
    selected = st.selectbox("Товар", options, index=default_index)
    product = products[products["option_label"] == selected].iloc[0]

    predictor = None
    lora_predictor = None
    if use_lora:
        try:
            lora_predictor = load_lora_predictor(LORA_DIR, HF_MODELS_DIR)
        except Exception as exc:
            st.error(f"Не удалось загрузить LoRA модели: {exc}")
            return
    else:
        try:
            predictor = load_predictor(MODEL_DIR, ARTIFACTS_DIR)
        except Exception as exc:
            st.error(f"Не удалось загрузить модель: {exc}")
            return

    if use_lora and lora_predictor is not None:
        preds = lora_predictor.predict(
            product["text"],
            product["good_type"],
            llm_reranker=reranker,
            rerank_top_n=20,
            rerank_max_output=5,
        )
    else:
        preds = predictor.predict(
            product["text"],
            product["good_type"],
            llm_reranker=reranker,
            rerank_top_n=20,
            rerank_max_output=5,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Информация о товаре")
        info_df = pd.DataFrame(
            [
                {
                    "SKU": product["product_sku"],
                    "Название": product.get("name", "—"),
                    "Категория": product.get("good_type", "—"),
                    "Подкатегория": product.get("good_subtype", "—"),
                }
            ]
        )
        st.table(info_df)

    with col2:
        st.subheader("Теги модели")
        if preds:
            pred_df = pd.DataFrame(preds, columns=["tag", "score"])
            pred_df["score"] = pred_df["score"].map(lambda x: f"{x:.3f}")
            st.table(pred_df)
        else:
            st.write("––")

    with st.expander("Отзывы"):
        for idx, comment in enumerate(product["comment_list"], start=1):
            st.markdown(f"**Отзыв {idx}:** {comment}")


if __name__ == "__main__":
    main()
