#!/usr/bin/env python3
"""
Script to sample up to MAX_ROWS_PER_TYPE rows for each unique good_type from lamoda_reviews.csv
Groups all reviews for each product into one row with reviews as an array
"""

import pandas as pd
from pathlib import Path
import json

MAX_ROWS_PER_TYPE = 10000

def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "lamoda_reviews.csv"
    output_file = project_root / "lamoda_reviews_sampled.csv"
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    print(f"Total rows: {len(df):,}")
    print(f"\nOriginal distribution by good_type:")
    print(df['good_type'].value_counts().sort_index())
    
    sampled_dfs = []
    for good_type in df['good_type'].unique():
        type_df = df[df['good_type'] == good_type]
        n_rows = len(type_df)
        
        if n_rows >= MAX_ROWS_PER_TYPE:
            sampled = type_df.sample(n=MAX_ROWS_PER_TYPE, random_state=42)
            print(f"{good_type}: sampled {MAX_ROWS_PER_TYPE:,} from {n_rows:,} rows")
        else:
            sampled = type_df
            print(f"{good_type}: using all {n_rows:,} rows (less than {MAX_ROWS_PER_TYPE:,} rows)")
        
        sampled_dfs.append(sampled)
    
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\nGrouping reviews by product...")
    grouped = result_df.groupby('product_sku').agg({
        'comment_id': list,
        'comment_text': list,
        'good_type': 'first',
        'good_subtype': 'first'
    }).reset_index()
    
    print(f"\nFiltering products with 4 or more reviews...")
    grouped['review_count'] = grouped['comment_text'].apply(len)
    filtered_grouped = grouped[grouped['review_count'] >= 4].copy()
    filtered_grouped = filtered_grouped.drop(columns=['review_count'])
    
    print(f"Products after filtering: {len(filtered_grouped):,} (removed {len(grouped) - len(filtered_grouped):,} products)")
    
    filtered_grouped['comment_id'] = filtered_grouped['comment_id'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    filtered_grouped['comment_text'] = filtered_grouped['comment_text'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    
    result_df = filtered_grouped
    
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal distribution by good_type:")
    print(result_df['good_type'].value_counts().sort_index())
    print(f"\nTotal rows in output: {len(result_df):,}")
    print(f"Unique products: {result_df['product_sku'].nunique():,}")
    
    print(f"\nSaving to {output_file}...")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print("Done!")


if __name__ == "__main__":
    main()
