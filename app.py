#!/usr/bin/env python3
"""
Simple Streamlit UI for exploring Lamoda products and model tags.
"""

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from model.data import build_text, parse_pylist_or_json, parse_tags_semicolon
from model.predict import TagPredictor

DATA_PATH = Path("lamoda_reviews_sampled_with_pull_tags.csv")
MODEL_DIR = Path("model/output/best_model")
ARTIFACTS_DIR = Path("model/artifacts")


@st.cache_data(show_spinner=False)
def load_products(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["comment_list"] = df["comment_text"].apply(parse_pylist_or_json)
    df["tag_list"] = df["tags"].apply(parse_tags_semicolon)
    df["text"] = df.apply(
        lambda r: build_text(
            r.get("name", ""),
            r.get("good_type", ""),
            r.get("good_subtype", ""),
            r["comment_list"],
        ),
        axis=1,
    )
    df["option_label"] = df.apply(
        lambda r: f"{r['product_sku']} — {r.get('name', 'Без названия')}", axis=1
    )
    return df


@st.cache_resource(show_spinner=False)
def load_predictor(model_dir: Path, artifacts_dir: Path) -> TagPredictor:
    return TagPredictor.load(model_dir, artifacts_dir, device="cpu")


def render_tag_list(title: str, tags: List[str]) -> None:
    st.markdown(f"**{title}**")
    if tags:
        st.write(", ".join(tags))
    else:
        st.write("—")


def main():
    st.set_page_config(page_title="Lamoda Tags Explorer", layout="wide")
    st.title("Lamoda Tags Explorer")
    st.caption("Выберите товар, просмотрите отзывы и посмотрите теги модели.")

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

    try:
        predictor = load_predictor(MODEL_DIR, ARTIFACTS_DIR)
    except Exception as exc:
        st.error(f"Не удалось загрузить модель: {exc}")
        return

    preds = predictor.predict(product["text"], product["good_type"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Информация о товаре")
        info_html = f"""
        <div style="border: 1px solid #e5e5e5; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <p><strong>SKU:</strong> {product['product_sku']}</p>
            <p><strong>Название:</strong> {product.get('name', '—')}</p>
            <p><strong>Категория:</strong> {product.get('good_type', '—')}</p>
            <p><strong>Подкатегория:</strong> {product.get('good_subtype', '—')}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
        render_tag_list("Теги из датасета", product["tag_list"])

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
