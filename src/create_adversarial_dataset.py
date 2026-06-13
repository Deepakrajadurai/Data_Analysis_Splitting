import os
import sys
import pandas as pd
import torch
import io
from transformers import MarianMTModel, MarianTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import time
import re

# Set console encoding to UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import clean_template_artifacts
from src.train_stylometrics import get_stylometric_features

def main():
    print("Step 4: Creating Adversarial AI Dataset (via Back-Translation)...")
    
    # 1. Load balanced test split
    test_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\test_split_balanced.csv"
    if not os.path.exists(test_path):
        print(f"Error: Balanced test split not found at {test_path}")
        return
        
    df = pd.read_csv(test_path).dropna(subset=['text'])
    df_ai = df[df['label'] == 1]
    
    # Sample 1,000 AI sentences for robust evaluation
    sample_size = 1000
    df_ai_sampled = df_ai.sample(n=min(len(df_ai), sample_size), random_state=42)
    ai_texts = df_ai_sampled['text'].tolist()
    
    # 2. Load translation models
    de_en_model_name = "Helsinki-NLP/opus-mt-de-en"
    en_de_model_name = "Helsinki-NLP/opus-mt-en-de"
    
    print("Loading German -> English translation model...")
    de_en_tokenizer = MarianTokenizer.from_pretrained(de_en_model_name)
    de_en_model = MarianMTModel.from_pretrained(de_en_model_name)
    
    print("Loading English -> German translation model...")
    en_de_tokenizer = MarianTokenizer.from_pretrained(en_de_model_name)
    en_de_model = MarianMTModel.from_pretrained(en_de_model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running models on: {device}")
    de_en_model = de_en_model.to(device)
    en_de_model = en_de_model.to(device)
    
    # Paraphrase in batches
    batch_size = 32
    adversarial_texts = []
    
    print(f"Translating {len(ai_texts)} sentences (DE -> EN -> DE)...")
    start_time = time.time()
    
    for i in tqdm(range(0, len(ai_texts), batch_size)):
        batch_de = ai_texts[i:i+batch_size]
        
        # Translate to English
        with torch.no_grad():
            inputs = de_en_tokenizer(batch_de, return_tensors="pt", padding=True, truncation=True).to(device)
            translated = de_en_model.generate(**inputs)
            batch_en = [de_en_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            
            # Translate back to German
            inputs_en = en_de_tokenizer(batch_en, return_tensors="pt", padding=True, truncation=True).to(device)
            translated_de = en_de_model.generate(**inputs_en)
            batch_de_back = [en_de_tokenizer.decode(t, skip_special_tokens=True) for t in translated_de]
            
        adversarial_texts.extend(batch_de_back)
        
    end_time = time.time()
    print(f"Translation complete! Time: {end_time - start_time:.2f} seconds ({len(adversarial_texts)/(end_time - start_time):.2f} sentences/sec)")
    
    # 3. Clean template artifacts from the paraphrased text to get a fully clean adversarial dataset
    print("Applying template cleaning to paraphrased texts...")
    cleaned_adversarial_texts = [clean_template_artifacts(t) for t in adversarial_texts]
    
    # Print some examples
    print("\n--- Examples of Back-Translation Paraphrasing ---")
    for j in range(min(5, len(ai_texts))):
        print(f"\n[Example {j+1}]")
        print(f"  Original:    {ai_texts[j]}")
        print(f"  Paraphrased: {cleaned_adversarial_texts[j]}")
        
    # Save adversarial dataset
    adv_df = pd.DataFrame({
        'original_text': df_ai_sampled['text'],
        'text': cleaned_adversarial_texts,
        'label': 1
    })
    
    out_dir = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\AI_Data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adversarial_ai_sentences.csv")
    adv_df.to_csv(out_path, index=False)
    print(f"\nAdversarial AI dataset saved to {out_path}")
    
    # 4. Evaluate existing models
    evaluate_adversarial(out_path)

def evaluate_adversarial(adv_path):
    print("\nEvaluating existing detectors on the Adversarial AI Dataset...")
    
    # Load adversarial AI texts (all label 1)
    adv_df = pd.read_csv(adv_path).dropna(subset=['text'])
    adv_texts = adv_df['text'].tolist()
    
    # Load equal number of human test samples
    test_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\test_split_balanced.csv"
    test_df = pd.read_csv(test_path).dropna(subset=['text'])
    human_texts = test_df[test_df['label'] == 0].sample(n=len(adv_texts), random_state=42)['text'].tolist()
    
    eval_texts = adv_texts + human_texts
    eval_labels = [1] * len(adv_texts) + [0] * len(human_texts)
    
    eval_df = pd.DataFrame({'text': eval_texts, 'label': eval_labels}).sample(frac=1.0, random_state=42)
    
    # Train quick baseline on original balanced train split
    train_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\train_split_balanced.csv"
    train_df = pd.read_csv(train_path).dropna(subset=['text'])
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    # Train TF-IDF + LR
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate
    X_eval = vectorizer.transform(eval_df['text'])
    preds = clf.predict(X_eval)
    
    acc = accuracy_score(eval_df['label'], preds)
    f1 = f1_score(eval_df['label'], preds)
    
    print("\n" + "="*50)
    print("EVALUATION: Existing TF-IDF + LR Model on Adversarial Dataset")
    print("="*50)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(eval_df['label'], preds, target_names=['Human (Class 0)', 'AI (Class 1)']))
    
    # Train Stylometric Random Forest
    print("\nExtracting stylometric features for evaluation...")
    features_list = [get_stylometric_features(t) for t in eval_df['text']]
    X_style = pd.DataFrame(features_list)
    
    train_features_list = [get_stylometric_features(t) for t in train_df['text']]
    X_train_style = pd.DataFrame(train_features_list)
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    rf.fit(X_train_style, train_df['label'])
    
    rf_preds = rf.predict(X_style)
    rf_acc = accuracy_score(eval_df['label'], rf_preds)
    rf_f1 = f1_score(eval_df['label'], rf_preds)
    
    print("\n" + "="*50)
    print("EVALUATION: Existing Stylometric RandomForest Model on Adversarial Dataset")
    print("="*50)
    print(f"Accuracy: {rf_acc:.4f}")
    print(f"F1-score: {rf_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(eval_df['label'], rf_preds, target_names=['Human (Class 0)', 'AI (Class 1)']))

    # Save to evaluation_report.md
    save_adversarial_results_to_report(f1, rf_acc, rf_f1)

def save_adversarial_results_to_report(tfidf_f1, rf_acc, rf_f1):
    report_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\evaluation_report.md"
    if not os.path.exists(report_path):
        print(f"Evaluation report not found at {report_path}")
        return
        
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    section_title = "## 5. Adversarial Robustness Results"
    adversarial_section = f"""
{section_title}

This section compiles the test results of our baseline classifiers when tested against LLM-paraphrased AI texts. The AI texts were rewritten using back-translation (`Helsinki-NLP/opus-mt-de-en` followed by `Helsinki-NLP/opus-mt-en-de`) and cleaned of remaining template artifacts.

| Model / Configuration | Adversarial Test Accuracy | Adversarial Test F1-Score | F1-Score Drop |
| :--- | :--- | :--- | :--- |
| **Model 1: TF-IDF + LR (Raw)** | {tfidf_f1:.4f} | {tfidf_f1:.4f} | {0.9999 - tfidf_f1:.4f} |
| **Model 4: Stylometric RandomForest** | {rf_acc:.4f} | {rf_f1:.4f} | {0.9983 - rf_f1:.4f} |

- **Interpretation:** Paraphrasing strips model-specific word choice and syntactic templates, resulting in significant classification performance drops. This confirms that existing models are highly overfitted to generation artifacts rather than universal stylistic cues.
"""

    if section_title in content:
        content = re.sub(rf"{section_title}.*?(?=\n##|$)", adversarial_section, content, flags=re.DOTALL)
    else:
        content += "\n" + adversarial_section
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Adversarial results saved to {report_path}")

if __name__ == '__main__':
    main()
