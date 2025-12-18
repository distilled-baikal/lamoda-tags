#!/usr/bin/env python3
"""
Generate tag pulls by good_type from lamoda reviews with tags.
Creates a set of unique lowercased tags for each good_type.
"""

import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "lamoda_reviews_sampled_with_tags.csv"
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    df = df[df['tags'].notna() & (df['tags'].str.strip() != '')]
    print(f"Rows with tags: {len(df)}")
    
    tag_pulls = {}
    
    for good_type, group in df.groupby('good_type'):
        all_tags = set()
        
        for tags_str in group['tags']:
            if pd.notna(tags_str) and tags_str.strip():
                tags = [tag.strip().lower() for tag in tags_str.split(';')]
                all_tags.update(t for t in tags if t)
        
        tag_pulls[good_type] = sorted(all_tags)
    
    print(f"\nTag pulls by good_type ({len(tag_pulls)} types):\n")
    for good_type, tags in sorted(tag_pulls.items()):
        print(f"=== {good_type} ({len(tags)} unique tags) ===")
        print(", ".join(tags))
        print()
    
    output_file = project_root / "tag_pulls_by_good_type.csv"
    rows = [{"good_type": gt, "tags": "; ".join(tags)} for gt, tags in sorted(tag_pulls.items())]
    pd.DataFrame(rows).to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()

