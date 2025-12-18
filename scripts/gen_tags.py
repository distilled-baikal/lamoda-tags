#!/usr/bin/env python3
"""
Script to generate tags for products based on their reviews using LLM.
Idempotent - resumes from where it left off if interrupted.
"""

import os
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
import json
from typing import List, Dict
import httpx
import warnings
import asyncio
from asyncio import Semaphore

# Disable SSL warnings for corporate proxy
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
warnings.filterwarnings('ignore', category=UserWarning, module='httpx')


# Configuration from mvp.ipynb
BASE_URL = os.getenv("OPENAI_BASE_URL", "")
API_KEY = os.getenv("OPENAI_TOKEN", "")
MODEL = os.getenv("OPENAI_MODEL", "")

# Chunking settings - approximate tokens (1 token ≈ 4 chars)
MAX_CHARS_PER_CHUNK = 600000  # Conservative estimate for ~2000 tokens


def chunk_reviews(reviews: List[str], max_chars: int = MAX_CHARS_PER_CHUNK) -> List[List[str]]:
    """Split reviews into chunks if they exceed max_chars."""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for review in reviews:
        review_size = len(review)
        
        if current_size + review_size > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [review]
            current_size = review_size
        else:
            current_chunk.append(review)
            current_size += review_size
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


async def call_llm(client: AsyncOpenAI, messages: List[Dict[str, str]], model: str = MODEL) -> str:
    """Call LLM API and return response - async version."""
    kwargs = {
        "model": model,
        "messages": messages,
    }
    
    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    return (choice.message.content or "").strip()


def extract_tags_from_response(response: str) -> str:
    """Extract tags from LLM JSON response - simple like mvp.ipynb."""
    import re
    import json
    
    if not response or not response.strip():
        return ""
    
    # Extract JSON from code blocks (like mvp.ipynb)
    pattern_json = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)
    matches = pattern_json.findall(response)
    
    if matches:
        json_text = matches[-1]
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_text = json_match.group(0)
        else:
            return ""
    
    try:
        data = json.loads(json_text)
        tags_list = data.get('теги', data.get('tags', []))
        
        if not isinstance(tags_list, list):
            return ""
        
        # Return all tags, joined by semicolon (no limit)
        tags = [str(tag).strip() for tag in tags_list if tag and str(tag).strip()]
        return '; '.join(tags)
        
    except json.JSONDecodeError:
        return ""


async def generate_tags_for_product(
    client: AsyncOpenAI,
    product_name: str,
    reviews: List[str],
    good_type: str,
    good_subtype: str
) -> str:
    """Generate tags for a product based on its reviews."""
    system_prompt = """Твоя задача - проанализировать отзывы покупателей и предложить краткие информативные теги, которые характеризуют товар.

ПРАВИЛА ФОРМУЛИРОВКИ ТЕГОВ:
1. Краткость: 1-3 слова максимум
2. Конкретность: описывают конкретные свойства товара (материал, качество, особенности использования)
3. Избегай абстракций: не используй общие фразы типа "популярный", "нравится", "хороший товар"
4. Избегай упоминания размера: не пиши "большой", "маленький", "в размер" (размер - это отдельная характеристика)
5. Нейтральный порядок слов: "Легко царапается" вместо "Царапается легко"
6. Противоречия: теги не должны противоречить друг другу
7. Количество: верни столько тегов, сколько нужно для полной характеристики товара (может быть любое количество)

ХОРОШИЕ ПРИМЕРЫ ТЕГОВ (конкретные, информативные):
- Качество/Материал: "Мягкая пена", "Плотная пена", "Качественная пена", "Мягкие", "Легкие"
- Функциональность: "Плохо пенится", "Заедает замок", "Плохо растегивается", "Нога не выскальзывает"
- Свойства: "Приятный запах", "Легкий запах", "Ненавязчивый аромат", "Универсальный запах"
- Особенности: "Пятка с мехом", "Для чувствительной кожи", "Справляется с потоотделением"
- Проблемы: "Сухая пена", "Не смягчает бритье"

ПЛОХИЕ ПРИМЕРЫ ТЕГОВ (избегай таких):
- Абстрактные: "Пользователям нравится", "Популярные", "Хороший товар", "Рекомендую", "Рекомендуем", "Нравится", "Любимый", "Такая прелесть", "Довольна покупкой", "Ребёнок доволен"
- Общие: "Качество", "Красиво", "Нормально", "Качественное качество", "Простенькая", "Кофточка", "Подстилка", "Коллекция", "Украшает", "Оригинал"
- Про размер/рост/фигуру: "Большой", "Маленький", "В размер", "Маломерят", "Большемерят", "Размер xl", "Рост 164", "Подходит на 42", "Подходит на 29", "Миниатюрные ноги", "Стройные ноги", "Классная талия"
- Про детали одежды/конструкции: "Подкладом", "Глубина декольте", "Порвал карман", "Перекашивает"
- Про цвет без контекста: "Черный цвет"
- Про несоответствие/отрицания: "Не подошел", "Не из замши", "Не оправданы"
- Про общие ощущения: "Носка комфортно"
- Мета-теги (ЗАПРЕЩЕНО): "Недостаточно данных", "Общий отзыв", "Без характеристик", "Нет информации", "Мало отзывов"

ВАЖНО:
- Если в отзывах недостаточно конкретной информации о товаре - НЕ генерируй теги вообще
- Если отзывы слишком общие (типа "нормально", "ок", "хорошо") - НЕ генерируй теги
- Генерируй теги ТОЛЬКО если есть конкретные характеристики товара в отзывах
- Если не можешь сгенерировать конкретные теги - верни пустой массив tags: []

ФОРМАТ ОТВЕТА:
Верни результат строго в единственном блоке кода формата ```json ... ``` без любого дополнительного текста.
Структура JSON:
{
  "explanation": "краткое объяснение (1-2 предложения) почему выбраны именно эти теги",
  "tags": ["тег1", "тег2", "тег3", ...]
}

Количество тегов может быть любым - добавляй столько, сколько нужно для полной характеристики товара.
Если нет конкретных характеристик - верни: {"explanation": "", "tags": []}

Важно: теги должны быть на русском языке и отражать реальные характеристики товара из отзывов."""
    
    # Prepare reviews text
    reviews_text = "\n".join([f"{i+1}. {review}" for i, review in enumerate(reviews)])
    
    user_prompt = f"""Товар: {product_name}
Тип: {good_type} / {good_subtype}

Отзывы покупателей:
{reviews_text}

Проанализируй отзывы и предложи теги для этого товара. Верни ответ в формате JSON в блоке кода:
```json
{{
  "explanation": "краткое объяснение",
  "tags": ["tag1", "tag2", "tag3", ...]
}}
```

Добавь столько тегов, сколько нужно для полной характеристики товара."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = await call_llm(client, messages)
    
    if not response:
        return ""
    
    tags = extract_tags_from_response(response)
    
    # Show tags in output
    if tags:
        print(f"  → Tags: {tags}")
    else:
        print(f"  → Warning: No tags extracted")
    
    return tags


async def generate_tags_for_product_chunked(
    client: AsyncOpenAI,
    product_name: str,
    reviews: List[str],
    good_type: str,
    good_subtype: str
) -> str:
    """Generate tags for a product when reviews need to be chunked."""
    chunks = chunk_reviews(reviews, MAX_CHARS_PER_CHUNK)
    
    if len(chunks) == 1:
        # Single chunk, use regular function
        return await generate_tags_for_product(client, product_name, reviews, good_type, good_subtype)
    
    # Multiple chunks - generate tags for each chunk, then combine
    all_tags = []
    
    for i, chunk in enumerate(chunks):
        chunk_tags = await generate_tags_for_product(
            client,
            f"{product_name} (часть {i+1}/{len(chunks)})",
            chunk,
            good_type,
            good_subtype
        )
        if chunk_tags:
            all_tags.extend(chunk_tags.split('; '))
    
    # Deduplicate tags (no limit on quantity)
    unique_tags = []
    seen = set()
    for tag in all_tags:
        tag_clean = tag.strip()
        if tag_clean and tag_clean.lower() not in seen:
            unique_tags.append(tag_clean)
            seen.add(tag_clean.lower())
    
    return '; '.join(unique_tags)


async def process_product(
    client: AsyncOpenAI,
    df: pd.DataFrame,
    product_row: pd.Series,
    results_dict: dict,
    lock: asyncio.Lock,
    semaphore: Semaphore,
    reset_failed: bool
) -> tuple:
    """Process a single product asynchronously. Returns (product_sku, tags)."""
    async with semaphore:
        product_sku = product_row['product_sku']
        
        # Skip if tags already exist for this product (and not failed)
        if not reset_failed:
            async with lock:
                product_tags = df[df['product_sku'] == product_sku]['tags']
                if not product_tags.empty:
                    first_tag = str(product_tags.iloc[0]) if pd.notna(product_tags.iloc[0]) else ''
                    if first_tag and first_tag.strip() and first_tag.strip() != '[FAILED]':
                        return (product_sku, None)
        
        # Get all reviews for this product
        # comment_text is now a JSON array string, need to parse it
        async with lock:
            product_row_data = df[df['product_sku'] == product_sku].iloc[0]
            comment_text_str = product_row_data['comment_text']
        
        # Parse JSON array from comment_text
        try:
            if pd.isna(comment_text_str) or not comment_text_str:
                return (product_sku, None)
            
            # Try to parse as JSON array
            reviews_list = json.loads(comment_text_str)
            if not isinstance(reviews_list, list):
                reviews_list = [comment_text_str]  # Fallback: treat as single review
        except (json.JSONDecodeError, TypeError):
            # If not JSON, treat as single review string
            reviews_list = [str(comment_text_str)] if comment_text_str else []
        
        # Filter out empty reviews
        reviews_list = [str(r).strip() for r in reviews_list if r and str(r).strip()]
        
        # Skip if no reviews
        if not reviews_list:
            return (product_sku, None)
        
        product_name = product_row['name']
        good_type = product_row['good_type']
        good_subtype = product_row['good_subtype']
        
        # Show product being processed
        print(f"\n{product_name} ({product_sku})")
        
        # Generate tags
        try:
            if len(' '.join(reviews_list)) > MAX_CHARS_PER_CHUNK:
                tags = await generate_tags_for_product_chunked(
                    client, product_name, reviews_list, good_type, good_subtype
                )
            else:
                tags = await generate_tags_for_product(
                    client, product_name, reviews_list, good_type, good_subtype
                )
            
            return (product_sku, tags)
            
        except Exception as e:
            print(f"\nError processing product {product_sku}: {e}")
            return (product_sku, None)


async def main_async():
    import sys
    
    # Check for test mode
    test_mode = '--test' in sys.argv
    reset_failed = '--reset-failed' in sys.argv
    max_concurrent = int(os.getenv("MAX_CONCURRENT", "2"))  # Max concurrent requests
    
    # Paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "lamoda_reviews_sampled.csv"
    output_file = project_root / "lamoda_reviews_sampled_with_tags.csv"
    
    # Load existing output file if it exists, otherwise start from input
    if output_file.exists():
        print(f"Loading existing data from {output_file}...")
        df = pd.read_csv(output_file, encoding='utf-8')
        if 'tags' not in df.columns:
            df['tags'] = ''
        # Replace NaN with empty string
        df['tags'] = df['tags'].fillna('')
        # Count products with valid tags (not empty and not failed)
        valid_tags = df[(df['tags'] != '') & (df['tags'] != '[FAILED]')]
        print(f"Resuming: {valid_tags['product_sku'].nunique()} products already have tags")
    else:
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file, encoding='utf-8')
        df['tags'] = ''
    
    # Initialize AsyncOpenAI client with SSL verification disabled (for corporate proxy)
    http_client = httpx.AsyncClient(verify=False)
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, http_client=http_client)
    
    # Get unique products
    products = df.groupby('product_sku').first().reset_index()
    
    # Reset failed tags if requested
    if reset_failed:
        df.loc[df['tags'] == '[FAILED]', 'tags'] = ''
        print("Reset all failed tags")
    
    # Limit to first 5 products in test mode
    if test_mode:
        products = products.head(5)
        print("TEST MODE: Processing only first 5 products")
    
    print(f"Processing {len(products)} products...")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Output will be saved to {output_file}")
    
    # Create semaphore to limit concurrent requests and lock for thread-safe DataFrame access
    semaphore = Semaphore(max_concurrent)
    lock = asyncio.Lock()
    
    # Process products in batches with progress tracking
    tasks = []
    for idx, product_row in products.iterrows():
        task = process_product(client, df, product_row, {}, lock, semaphore, reset_failed)
        tasks.append(task)
    
    # Process all tasks with progress bar
    results = await async_tqdm.gather(*tasks, desc="Generating tags")
    
    # Update DataFrame with results
    for product_sku, tags in results:
        if tags is not None:
            df.loc[df['product_sku'] == product_sku, 'tags'] = tags
    
    # Save final results with UTF-8 encoding for Russian text
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Close HTTP client
    await http_client.aclose()
    
    print(f"\nDone! Results saved to {output_file}")
    print(f"\nTag statistics:")
    tagged_count = (df['tags'] != '').sum()
    print(f"Products with tags: {tagged_count}/{len(products)}")


def main():
    """Main entry point - runs async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

