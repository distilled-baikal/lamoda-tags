#!/usr/bin/env python3
"""
Generate best tags from raw tag pulls using LLM and embeddings.
- Uses embeddings to find similar tags
- Uses LLM to consolidate and select best tags
- Saves mapping: old_tag1, old_tag2 -> new_tag
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from openai import AsyncOpenAI
import json
from typing import List, Dict, Tuple
import httpx
import warnings
import asyncio
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
warnings.filterwarnings('ignore', category=UserWarning, module='httpx')

BASE_URL = os.getenv("OPENAI_BASE_URL", "")
API_KEY = os.getenv("OPENAI_TOKEN", "")
MODEL = os.getenv("OPENAI_MODEL", "")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b")

SIMILARITY_THRESHOLD = 0.85


async def get_embeddings(client: AsyncOpenAI, texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Get embeddings for a list of texts."""
    if not texts:
        return np.array([])
    
    resp = await client.embeddings.create(input=texts, model=model)
    embeddings = [item.embedding for item in resp.data]
    return np.array(embeddings)


def cluster_similar_tags(tags: List[str], embeddings: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> List[List[str]]:
    """Cluster similar tags using embeddings."""
    if len(tags) <= 1:
        return [[t] for t in tags]
    
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - sim_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    clusters = {}
    for tag, label in zip(tags, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(tag)
    
    return list(clusters.values())


async def call_llm(client: AsyncOpenAI, messages: List[Dict[str, str]], model: str = MODEL) -> str:
    """Call LLM API and return response."""
    resp = await client.chat.completions.create(model=model, messages=messages)
    return (resp.choices[0].message.content or "").strip()


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    import re
    
    if not response or not response.strip():
        return {}
    
    pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = pattern.findall(response)
    
    if matches:
        json_text = matches[-1]
    else:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_text = json_match.group(0)
        else:
            return {}
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {}


async def consolidate_tag_cluster(client: AsyncOpenAI, cluster: List[str], good_type: str) -> Tuple[str, List[str]]:
    """Use LLM to consolidate a cluster of similar tags into one best tag."""
    if len(cluster) == 1:
        return cluster[0], cluster
    
    system_prompt = """Ты эксперт по созданию информативных тегов для товаров.
Тебе даётся кластер похожих тегов. Твоя задача - выбрать или создать ОДИН лучший тег, который:
1. Краткий (1-3 слова)
2. Конкретный и информативный
3. Объединяет смысл всех тегов в кластере
4. НЕ является абстрактным ("хороший", "качественный" без контекста)

Если теги в кластере слишком разные по смыслу - верни пустую строку.
Если теги бессмысленные или слишком общие - верни пустую строку.

ФОРМАТ ОТВЕТА (только JSON):
```json
{
  "best_tag": "выбранный или созданный тег",
  "reason": "почему этот тег лучший"
}
```

Если нельзя объединить - верни: {"best_tag": "", "reason": "причина"}"""

    user_prompt = f"""Тип товара: {good_type}

Кластер похожих тегов:
{json.dumps(cluster, ensure_ascii=False, indent=2)}

Выбери или создай один лучший тег для этого кластера."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = await call_llm(client, messages)
    data = extract_json_from_response(response)
    best_tag = data.get("best_tag", "").strip()
    
    return best_tag, cluster


async def filter_and_select_best_tags(
    client: AsyncOpenAI,
    tags: List[str],
    good_type: str
) -> Tuple[List[str], List[Tuple[List[str], str]]]:
    """Filter bad tags and select best ones using LLM."""
    
    system_prompt = """Ты эксперт по фильтрации тегов для товаров интернет-магазина.

Тебе даётся список тегов для определённого типа товара. Твоя задача:
1. УДАЛИТЬ плохие теги:
   - Абстрактные: "хороший", "качественный", "нравится", "рекомендую"
   - Про размер: "маломерит", "большемерит", "в размер"
   - Общие: "нормально", "ок", "супер"
   - Бессмысленные или слишком специфичные
   
2. ОБЪЕДИНИТЬ похожие теги (синонимы) в один лучший:
   - "быстро сохнет", "сохнет быстро" -> "быстро сохнет"
   - "хорошо впитывает", "отлично впитывает", "впитывает хорошо" -> "хорошо впитывает"
   - "мягкий материал", "мягкая ткань", "мягкие" -> "мягкий"
   
3. ОСТАВИТЬ только информативные теги, описывающие конкретные свойства товара

ФОРМАТ ОТВЕТА (только JSON):
```json
{
  "kept_tags": ["тег1", "тег2", ...],
  "mappings": [
    {"old": ["старый1", "старый2"], "new": "новый"},
    {"old": ["удалённый1"], "new": ""}
  ]
}
```

mappings должен содержать ВСЕ исходные теги - либо объединённые в новый, либо удалённые (new: "")."""

    user_prompt = f"""Тип товара: {good_type}

Список тегов ({len(tags)} шт.):
{json.dumps(tags, ensure_ascii=False, indent=2)}

Отфильтруй и объедини теги."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = await call_llm(client, messages)
    data = extract_json_from_response(response)
    
    kept_tags = data.get("kept_tags", [])
    raw_mappings = data.get("mappings", [])
    
    mappings = []
    for m in raw_mappings:
        old_tags = m.get("old", [])
        new_tag = m.get("new", "")
        if old_tags:
            mappings.append((old_tags, new_tag))
    
    return kept_tags, mappings


async def process_good_type(
    client: AsyncOpenAI,
    good_type: str,
    tags: List[str]
) -> Tuple[List[str], List[Tuple[List[str], str]]]:
    """Process tags for a single good_type."""
    print(f"\n=== {good_type} ({len(tags)} tags) ===")
    
    print("  Getting embeddings...")
    embeddings = await get_embeddings(client, tags)
    
    print("  Clustering similar tags...")
    clusters = cluster_similar_tags(tags, embeddings)
    print(f"  Found {len(clusters)} clusters")
    
    print("  Consolidating clusters...")
    consolidated_tags = []
    cluster_mappings = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            consolidated_tags.append(cluster[0])
        else:
            best_tag, original_tags = await consolidate_tag_cluster(client, cluster, good_type)
            if best_tag:
                consolidated_tags.append(best_tag)
                if best_tag not in original_tags:
                    cluster_mappings.append((original_tags, best_tag))
                else:
                    others = [t for t in original_tags if t != best_tag]
                    if others:
                        cluster_mappings.append((others, best_tag))
    
    print(f"  After clustering: {len(consolidated_tags)} tags")
    
    print("  Final filtering...")
    final_tags, llm_mappings = await filter_and_select_best_tags(client, consolidated_tags, good_type)
    
    all_mappings = cluster_mappings + llm_mappings
    
    print(f"  Final: {len(final_tags)} tags")
    print(f"  Mappings: {len(all_mappings)}")
    
    return final_tags, all_mappings


async def main_async():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "tag_pulls_raw.csv"
    output_tags_file = project_root / "tag_pulls_best.csv"
    output_mappings_file = project_root / "tag_mappings.csv"
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    http_client = httpx.AsyncClient(verify=False)
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=http_client)
    
    all_best_tags = {}
    all_mappings = []
    
    for _, row in df.iterrows():
        good_type = row['good_type']
        tags_str = row['tags']
        
        if pd.isna(tags_str) or not tags_str.strip():
            continue
        
        tags = [t.strip() for t in tags_str.split(';') if t.strip()]
        
        if not tags:
            continue
        
        best_tags, mappings = await process_good_type(client, good_type, tags)
        all_best_tags[good_type] = best_tags
        
        for old_tags, new_tag in mappings:
            all_mappings.append({
                "good_type": good_type,
                "old_tags": "; ".join(old_tags),
                "new_tag": new_tag
            })
    
    await http_client.aclose()
    
    rows = [{"good_type": gt, "tags": "; ".join(tags)} for gt, tags in sorted(all_best_tags.items())]
    pd.DataFrame(rows).to_csv(output_tags_file, index=False, encoding='utf-8')
    print(f"\nSaved best tags to {output_tags_file}")
    
    pd.DataFrame(all_mappings).to_csv(output_mappings_file, index=False, encoding='utf-8')
    print(f"Saved mappings to {output_mappings_file}")
    
    print("\n=== Summary ===")
    for gt, tags in sorted(all_best_tags.items()):
        original_count = len([t.strip() for t in df[df['good_type'] == gt]['tags'].iloc[0].split(';') if t.strip()])
        print(f"{gt}: {original_count} -> {len(tags)} tags")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

