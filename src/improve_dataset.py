import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import clean_template_artifacts

def length_balanced_sampling(df, bin_size=20, random_state=42):
    """
    Groups data by character length bins, and samples an equal number
    of Human (label 0) and AI (label 1) sentences from each bin.
    """
    # Calculate lengths
    df = df.copy()
    df['char_len'] = df['text'].str.len()
    
    # Define bins
    max_len = int(df['char_len'].max()) + 1
    bins = list(range(0, max_len + bin_size, bin_size))
    
    # Assign each record to a bin
    df['bin'] = pd.cut(df['char_len'], bins=bins, labels=False)
    
    balanced_chunks = []
    
    # Loop through bins and balance each one
    for bin_idx, group in df.groupby('bin'):
        human_group = group[group['label'] == 0]
        ai_group = group[group['label'] == 1]
        
        n_samples = min(len(human_group), len(ai_group))
        if n_samples > 0:
            sampled_human = human_group.sample(n=n_samples, random_state=random_state)
            sampled_ai = ai_group.sample(n=n_samples, random_state=random_state)
            balanced_chunks.append(sampled_human)
            balanced_chunks.append(sampled_ai)
            
    if not balanced_chunks:
        return pd.DataFrame(columns=df.columns)
        
    balanced_df = pd.concat(balanced_chunks).sample(frac=1.0, random_state=random_state)
    return balanced_df

def improve_split(in_path, out_path):
    print(f"\nImproving dataset split: {in_path} -> {out_path}")
    df = pd.read_csv(in_path).dropna(subset=['text'])
    
    # 1. Clean the text of all template leaks
    print("Applying advanced template cleaning...")
    df['text'] = df['text'].apply(clean_template_artifacts)
    
    # Drop rows that became empty or too short after cleaning
    df = df[df['text'].str.strip().str.len() > 10]
    
    # Print average lengths before balancing
    human_before = df[df['label'] == 0]
    ai_before = df[df['label'] == 1]
    print("Before length balancing:")
    print(f"  Human count: {len(human_before)}, Avg char length: {human_before['text'].str.len().mean():.2f}")
    print(f"  AI count:    {len(ai_before)}, Avg char length: {ai_before['text'].str.len().mean():.2f}")
    
    # 2. Apply length balancing
    print("Applying length balancing...")
    balanced_df = length_balanced_sampling(df)
    
    # Print average lengths after balancing
    human_after = balanced_df[balanced_df['label'] == 0]
    ai_after = balanced_df[balanced_df['label'] == 1]
    print("After length balancing:")
    print(f"  Human count: {len(human_after)}, Avg char length: {human_after['text'].str.len().mean():.2f}")
    print(f"  AI count:    {len(ai_after)}, Avg char length: {ai_after['text'].str.len().mean():.2f}")
    
    # Drop temp columns and save
    balanced_df = balanced_df.drop(columns=['char_len', 'bin'])
    balanced_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path} (Total size: {len(balanced_df)} rows)")

def main():
    print("Step 3: Improving the Dataset Splits...")
    
    base_dir = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data"
    
    train_in = os.path.join(base_dir, "train_split_balanced.csv")
    val_in = os.path.join(base_dir, "val_split_balanced.csv")
    test_in = os.path.join(base_dir, "test_split_balanced.csv")
    
    train_out = os.path.join(base_dir, "train_split_improved.csv")
    val_out = os.path.join(base_dir, "val_split_improved.csv")
    test_out = os.path.join(base_dir, "test_split_improved.csv")
    
    improve_split(train_in, train_out)
    improve_split(val_in, val_out)
    improve_split(test_in, test_out)
    
    print("\nDataset improvement complete!")

if __name__ == '__main__':
    main()
