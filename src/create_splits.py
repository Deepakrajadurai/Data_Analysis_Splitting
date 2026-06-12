import os
import sys
import hashlib
import csv
from tqdm import tqdm
import random

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_ai_data_generator, load_human_data_generator

def get_split(group_key):
    """
    Deterministically assigns a group key to a split:
    - 0-79: train (80%)
    - 80-89: val (10%)
    - 90-99: test (10%)
    """
    h = hashlib.md5(group_key.encode('utf-8')).hexdigest()
    val = int(h, 16) % 100
    if val < 80:
        return 'train'
    elif val < 90:
        return 'val'
    else:
        return 'test'

def run_splitting(subsample_size=None):
    print("Phase 2: Creating Proper Splits (Train 80%, Val 10%, Test 10%)...")
    
    human_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\model_ready_dataset.csv"
    ai_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\AI_Data\ai_generated_texts.jsonl"
    
    # Store split buckets
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # Track groups for verification
    group_assignments = {}
    
    # Load Human data
    print("Splitting Human data...")
    human_gen = load_human_data_generator(human_path)
    for record in tqdm(human_gen, desc="Human Records", total=2074797):
        domain = record['domain']
        doc_id = record['document_id']
        spk = record['speaker']
        
        # Define group key
        if domain == 'debate':
            group_key = f"debate_spk_{spk if spk else 'unknown'}"
        else: # legal
            group_key = f"legal_doc_{doc_id if doc_id else 'unknown'}"
            
        split_name = get_split(group_key)
        splits[split_name].append(record)
        
        # Log group for leakage check
        if group_key in group_assignments and group_assignments[group_key] != split_name:
            print(f"ERROR: Leakage detected for group {group_key}!")
        group_assignments[group_key] = split_name

    # Load AI data
    print("Splitting AI data...")
    ai_gen = load_ai_data_generator(ai_path)
    for record in tqdm(ai_gen, desc="AI Records", total=250001):
        # AI texts have unique document_ids and no speakers.
        # We group by their unique document_id to distribute them across splits.
        doc_id = record['document_id']
        group_key = f"ai_doc_{doc_id}"
        
        split_name = get_split(group_key)
        splits[split_name].append(record)
        
        group_assignments[group_key] = split_name

    # 1. Output Split Stats
    print("\n--- Split Size Statistics (Full Dataset) ---")
    for s_name in ['train', 'val', 'test']:
        total = len(splits[s_name])
        human_cnt = sum(1 for r in splits[s_name] if r['label'] == 0)
        ai_cnt = sum(1 for r in splits[s_name] if r['label'] == 1)
        print(f"  {s_name.upper()}: {total:,} rows (Human: {human_cnt:,}, AI: {ai_cnt:,})")
        
    # 2. Verify splits leakages
    print("\nVerifying constraints...")
    train_groups = {g for g, s in group_assignments.items() if s == 'train'}
    val_groups = {g for g, s in group_assignments.items() if s == 'val'}
    test_groups = {g for g, s in group_assignments.items() if s == 'test'}
    
    print(f"  Overlap Train-Val: {len(train_groups.intersection(val_groups))}")
    print(f"  Overlap Train-Test: {len(train_groups.intersection(test_groups))}")
    print(f"  Overlap Val-Test: {len(val_groups.intersection(test_groups))}")
    assert len(train_groups.intersection(val_groups)) == 0, "Train-Val Leakage!"
    assert len(train_groups.intersection(test_groups)) == 0, "Train-Test Leakage!"
    assert len(val_groups.intersection(test_groups)) == 0, "Val-Test Leakage!"
    print("  Constraint verification PASSED! Zero group leakage.")

    # 3. Save full CSVs
    output_dir = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data"
    fieldnames = ['text', 'label', 'domain', 'source_type', 'document_id', 'speaker', 'url']
    
    for s_name in ['train', 'val', 'test']:
        out_path = os.path.join(output_dir, f"{s_name}_split_full.csv")
        print(f"Saving full {s_name} split to {out_path}...")
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in splits[s_name]:
                writer.writerow(record)

    # 4. Generate Balanced Downsampled Splits (Subsampling for faster baseline training)
    # Default to subsample if specified, e.g. Train: 100k per class (total 200k), Val: 10k per class, Test: 10k per class
    print("\nGenerating downsampled balanced splits...")
    sub_sizes = {
        'train': 100000,
        'val': 10000,
        'test': 10000
    }
    
    for s_name in ['train', 'val', 'test']:
        # Separate classes
        human_recs = [r for r in splits[s_name] if r['label'] == 0]
        ai_recs = [r for r in splits[s_name] if r['label'] == 1]
        
        target_size = sub_sizes[s_name]
        # Shuffle deterministically to make it reproducible
        random.seed(42)
        random.shuffle(human_recs)
        random.shuffle(ai_recs)
        
        sub_human = human_recs[:target_size]
        sub_ai = ai_recs[:target_size]
        
        sub_all = sub_human + sub_ai
        random.shuffle(sub_all)
        
        out_path = os.path.join(output_dir, f"{s_name}_split_balanced.csv")
        print(f"Saving downsampled balanced {s_name} split ({len(sub_all):,} rows) to {out_path}...")
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in sub_all:
                writer.writerow(record)

if __name__ == '__main__':
    run_splitting()
