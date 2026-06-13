import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_ai_data_generator, load_human_data_generator, clean_text, TEMPLATE_KEYWORDS

def get_model_family(model_name):
    model_name_lower = str(model_name).lower()
    if 'gemini' in model_name_lower:
        return 'Gemini'
    elif 'phi3' in model_name_lower:
        return 'Phi3'
    elif 'mistral' in model_name_lower or 'mixtral' in model_name_lower:
        return 'Mistral'
    elif 'gemma' in model_name_lower:
        return 'Gemma'
    elif 'llama' in model_name_lower:
        return 'Llama'
    else:
        return 'Other'

def to_markdown_table(df):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)

def run_lomo():
    print("Step 2: Running Leave-One-Model-Out (LOMO) Experiments...")
    
    # 1. Load all AI data with model families
    ai_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\AI_Data\ai_generated_texts.jsonl"
    print("Loading AI dataset...")
    ai_records = []
    for rec in load_ai_data_generator(ai_path):
        model_name = rec['speaker']
        family = get_model_family(model_name)
        rec['family'] = family
        ai_records.append(rec)
        
    ai_df = pd.DataFrame(ai_records)
    print(f"Total AI records loaded: {len(ai_df)}")
    print("Distribution of AI records by model family:")
    print(ai_df['family'].value_counts())
    
    # 2. Load all Human data
    human_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\model_ready_dataset.csv"
    print("\nLoading Human dataset...")
    # Load all human records as a list of dicts first to avoid massive memory footprint of pandas loading 2M rows if we only need a sample
    human_records = []
    human_gen = load_human_data_generator(human_path)
    # Since we need a good amount of human data, we load up to 250k human sentences to match AI total size
    for idx, rec in enumerate(human_gen):
        human_records.append(rec)
        if idx >= 300000:  # limit to speed up loading and keep memory low
            break
            
    human_df = pd.DataFrame(human_records)
    print(f"Total Human records loaded for sampling pool: {len(human_df)}")
    
    # We will use cleaned text and exclude custom stopwords to measure real generalization
    stop_words = list(TEMPLATE_KEYWORDS)
    
    experiments = [
        {
            'train_families': ['Gemini', 'Phi3', 'Mistral'],
            'test_family': 'Llama'
        },
        {
            'train_families': ['Gemini', 'Llama', 'Mistral'],
            'test_family': 'Phi3'
        },
        {
            'train_families': ['Gemini', 'Phi3', 'Mistral', 'Llama'], # All except Gemma
            'test_family': 'Gemma'
        }
    ]
    
    results = []
    
    for exp in experiments:
        train_fams = exp['train_families']
        test_fam = exp['test_family']
        print(f"\n--- Running Experiment: Train on {train_fams} -> Test on {test_fam} ---")
        
        # Train AI set
        train_ai = ai_df[ai_df['family'].isin(train_fams)]
        # Test AI set
        test_ai = ai_df[ai_df['family'] == test_fam]
        
        n_train_ai = len(train_ai)
        n_test_ai = len(test_ai)
        
        print(f"Train AI count: {n_train_ai}, Test AI count: {n_test_ai}")
        
        # Sample balanced human data
        train_human = human_df.sample(n=n_train_ai, random_state=42)
        # Test human data (ensure no overlap with train human)
        test_human_pool = human_df.drop(train_human.index)
        test_human = test_human_pool.sample(n=n_test_ai, random_state=42)
        
        # Combine
        train_df = pd.concat([
            pd.DataFrame({'text': train_ai['text'], 'label': 1}),
            pd.DataFrame({'text': train_human['text'], 'label': 0})
        ]).sample(frac=1.0, random_state=42)
        
        test_df = pd.concat([
            pd.DataFrame({'text': test_ai['text'], 'label': 1}),
            pd.DataFrame({'text': test_human['text'], 'label': 0})
        ]).sample(frac=1.0, random_state=42)
        
        # Clean texts
        print("Cleaning text data...")
        train_df['cleaned_text'] = train_df['text'].apply(clean_text)
        test_df['cleaned_text'] = test_df['text'].apply(clean_text)
        
        # Vectorize
        print("Vectorizing...")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words=stop_words)
        X_train = vectorizer.fit_transform(train_df['cleaned_text'])
        y_train = train_df['label']
        
        X_test = vectorizer.transform(test_df['cleaned_text'])
        y_test = test_df['label']
        
        # Train
        print("Training Logistic Regression...")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Predict & Evaluate
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        print(f"Results: Accuracy = {acc:.4f}, F1-score = {f1:.4f}")
        print(classification_report(y_test, preds, target_names=['Human (Class 0)', 'AI (Class 1)']))
        
        results.append({
            'Train Models': "+".join(train_fams),
            'Test Model': test_fam,
            'Accuracy': f"{acc*100:.2f}%",
            'F1': f"{f1:.4f}"
        })
        
    print("\n" + "="*50)
    print("LEAVE-ONE-MODEL-OUT EXPERIMENTS FINAL RESULTS TABLE")
    print("="*50)
    res_df = pd.DataFrame(results)
    print(to_markdown_table(res_df))
    
    # Save the table in evaluation_report.md
    save_results_to_report(res_df)

def save_results_to_report(res_df):
    report_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\evaluation_report.md"
    if not os.path.exists(report_path):
        print(f"Evaluation report not found at {report_path}")
        return
        
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find or create a section for LOMO
    markdown_table = to_markdown_table(res_df)
    section_title = "## 4. Leave-One-Model-Out (LOMO) Experiment Results"
    
    lomo_section = f"""
{section_title}

This experiment evaluates the detector's capability to identify AI text generated by a model family that was completely omitted from the training set. This tests whether the model learns generalizable features of machine-generated text or simply memorizes model-specific generation artifacts.

{markdown_table}

- **Interpretation:** If the F1-score remains high (> 0.85) on unseen models, it indicates that the detector is highly generalizable and capturing fundamental structural/stylistic differences of AI generation.
"""
    
    if section_title in content:
        # Replace existing section
        content = re.sub(rf"{section_title}.*?(?=\n##|$)", lomo_section, content, flags=re.DOTALL)
    else:
        # Append to the end
        content += "\n" + lomo_section
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"LOMO results saved to {report_path}")

if __name__ == '__main__':
    run_lomo()
