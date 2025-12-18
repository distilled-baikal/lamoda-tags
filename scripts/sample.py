#!/usr/bin/env python3
"""
Script to sample up to MAX_ROWS_PER_TYPE rows for each unique good_type from lamoda_reviews.csv
"""

import pandas as pd
from pathlib import Path

MAX_ROWS_PER_TYPE = 1000

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "lamoda_reviews.csv"
    output_file = project_root / "lamoda_reviews_sampled.csv"
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Total rows: {len(df):,}")
    print(f"\nOriginal distribution by good_type:")
    print(df['good_type'].value_counts().sort_index())
    
    # Filter to only include products with more than 4 reviews
    print(f"\nFiltering products with more than 4 reviews...")
    review_counts = df['product_sku'].value_counts()
    products_with_many_reviews = review_counts[review_counts > 4].index
    filtered_df = df[df['product_sku'].isin(products_with_many_reviews)]
    
    print(f"Rows after filtering: {len(filtered_df):,} (removed {len(df) - len(filtered_df):,} rows)")
    print(f"\nFiltered distribution by good_type:")
    print(filtered_df['good_type'].value_counts().sort_index())
    
    # Sample up to 5k rows for each good_type
    sampled_dfs = []
    for good_type in filtered_df['good_type'].unique():
        type_df = filtered_df[filtered_df['good_type'] == good_type]
        n_rows = len(type_df)
        
        if n_rows >= MAX_ROWS_PER_TYPE:
            sampled = type_df.sample(n=MAX_ROWS_PER_TYPE, random_state=42)
            print(f"{good_type}: sampled {MAX_ROWS_PER_TYPE:,} from {n_rows:,} rows")
        else:
            sampled = type_df
            print(f"{good_type}: using all {n_rows:,} rows (less than {MAX_ROWS_PER_TYPE:,} rows)")
        
        sampled_dfs.append(sampled)
    
    # Combine all sampled dataframes
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the final dataframe
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal distribution by good_type:")
    print(result_df['good_type'].value_counts().sort_index())
    print(f"\nTotal rows in output: {len(result_df):,}")
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    result_df.to_csv(output_file, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
