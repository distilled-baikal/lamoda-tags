#!/usr/bin/env python3
"""
Обновляет теги в отзывах Lamoda, заменяя их на теги из пула.
Использует маппинг из tag_mappings.csv для замены старых тегов на новые,
и оставляет только те теги, которые есть в tag_pulls_best.csv.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Set, List
from tqdm import tqdm


def load_mappings(mappings_file: Path) -> Dict[str, Dict[str, str]]:
    """Загружает маппинг тегов: good_type -> {old_tag: new_tag}."""
    print(f"Loading mappings from {mappings_file}...")
    df = pd.read_csv(mappings_file, encoding='utf-8')
    
    mappings = {}
    for _, row in df.iterrows():
        good_type = row['good_type']
        old_tags_str = row['old_tags']
        new_tag_raw = row['new_tag']
        
        if pd.isna(new_tag_raw):
            new_tag = ""
        else:
            new_tag = str(new_tag_raw).strip().lower()
        
        if pd.isna(old_tags_str):
            continue
        
        if good_type not in mappings:
            mappings[good_type] = {}
        
        old_tags = [t.strip().lower() for t in str(old_tags_str).split(';') if t.strip()]
        
        for old_tag in old_tags:
            mappings[good_type][old_tag] = new_tag
    
    print(f"Loaded {sum(len(m) for m in mappings.values())} mappings for {len(mappings)} good types")
    return mappings


def load_best_tags(best_tags_file: Path) -> Dict[str, Set[str]]:
    """Загружает лучшие теги для каждого good_type."""
    print(f"Loading best tags from {best_tags_file}...")
    df = pd.read_csv(best_tags_file, encoding='utf-8')
    
    best_tags = {}
    for _, row in df.iterrows():
        good_type = row['good_type']
        tags_str = row['tags']
        
        tags = [t.strip().lower() for t in tags_str.split(';') if t.strip()]
        best_tags[good_type] = set(tags)
    
    print(f"Loaded best tags for {len(best_tags)} good types")
    return best_tags


def apply_mapping_and_filter(
    tags: List[str],
    good_type: str,
    mappings: Dict[str, Dict[str, str]],
    best_tags: Dict[str, Set[str]]
) -> List[str]:
    """Применяет маппинг и фильтрует теги, оставляя только из пула."""
    if not tags:
        return []
    
    normalized_tags = [t.strip().lower() for t in tags if t.strip()]
    
    mapped_tags = []
    type_mappings = mappings.get(good_type, {})
    
    for tag in normalized_tags:
        if tag in type_mappings:
            new_tag = type_mappings[tag]
            if new_tag:
                mapped_tags.append(new_tag)
        else:
            mapped_tags.append(tag)
    
    type_best_tags = best_tags.get(good_type, set())
    filtered_tags = [tag for tag in mapped_tags if tag in type_best_tags]
    
    seen = set()
    unique_tags = []
    for tag in filtered_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    
    return unique_tags


def main():
    project_root = Path(__file__).parent.parent
    
    input_file = project_root / "lamoda_reviews_sampled_with_tags.csv"
    mappings_file = project_root / "tag_mappings.csv"
    best_tags_file = project_root / "tag_pulls_best.csv"
    output_file = project_root / "lamoda_reviews_sampled_with_pull_tags.csv"
    
    print(f"Reading reviews from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"Loaded {len(df)} reviews")
    
    mappings = load_mappings(mappings_file)
    best_tags = load_best_tags(best_tags_file)
    
    print("\nUpdating tags...")
    updated_tags = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        good_type = row['good_type']
        tags_str = row['tags']
        
        if pd.isna(tags_str) or not tags_str.strip():
            updated_tags.append("")
        else:
            tags = [t.strip() for t in tags_str.split(';') if t.strip()]
            new_tags = apply_mapping_and_filter(tags, good_type, mappings, best_tags)
            updated_tags.append("; ".join(new_tags))
    
    df_output = df.copy()
    df_output['tags'] = updated_tags
    
    print(f"\nSaving updated reviews to {output_file}...")
    df_output.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ Saved {len(df_output)} reviews to {output_file}")
    
    print("\n=== Summary ===")
    original_with_tags = sum(1 for tags_str in df['tags'] if pd.notna(tags_str) and tags_str.strip())
    updated_with_tags = sum(1 for tags_str in updated_tags if tags_str.strip())
    
    print(f"Reviews with tags (original): {original_with_tags}")
    print(f"Reviews with tags (updated): {updated_with_tags}")
    
    original_tag_count = sum(
        len([t for t in tags_str.split(';') if t.strip()])
        for tags_str in df['tags']
        if pd.notna(tags_str) and tags_str.strip()
    )
    updated_tag_count = sum(
        len([t for t in tags_str.split(';') if t.strip()])
        for tags_str in updated_tags
        if tags_str.strip()
    )
    
    print(f"Total tags (original): {original_tag_count}")
    print(f"Total tags (updated): {updated_tag_count}")
    print(f"Reduction: {original_tag_count - updated_tag_count} tags ({100 * (1 - updated_tag_count / original_tag_count):.1f}%)")


if __name__ == "__main__":
    main()

