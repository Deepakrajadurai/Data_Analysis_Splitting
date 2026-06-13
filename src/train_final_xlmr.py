import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(filepath, sample_size=None):
    print(f"Loading data from {filepath} (sample_size={sample_size})...")
    df = pd.read_csv(filepath).dropna(subset=['text'])
    
    if sample_size is not None:
        # Balanced downsampling
        df_human = df[df['label'] == 0]
        df_ai = df[df['label'] == 1]
        
        n_each = sample_size // 2
        df_human_sampled = df_human.sample(n=min(len(df_human), n_each), random_state=42)
        df_ai_sampled = df_ai.sample(n=min(len(df_ai), n_each), random_state=42)
        df = pd.concat([df_human_sampled, df_ai_sampled]).sample(frac=1.0, random_state=42)
        
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def main():
    print("Phase 2: Training Final XLM-R Model...")
    
    train_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\train_split_improved.csv"
    test_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\test_split_improved.csv"
    
    # 1. Load Data
    # Train: 10,000 sentences (5,000 human, 5,000 AI)
    train_texts, train_labels = load_data(train_path, sample_size=10000)
    # Test: 2,000 sentences (1,000 human, 1,000 AI)
    test_texts, test_labels = load_data(test_path, sample_size=2000)
    
    # 2. Setup model and tokenizer
    model_name = "xlm-roberta-base"
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # 3. Create datasets and loaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 4. Training
    print("Fine-tuning XLM-RoBERTa on CPU...")
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc="Training")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    print(f"Training Epoch Loss: {total_loss/len(train_loader):.4f}")
    
    # 5. Evaluation & Misclassification Extraction
    print("Running final evaluation...")
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            test_preds.extend(preds)
            test_targets.extend(labels.numpy())
            
    # Calculate Metrics
    acc = accuracy_score(test_targets, test_preds)
    prec = precision_score(test_targets, test_preds)
    rec = recall_score(test_targets, test_preds)
    f1 = f1_score(test_targets, test_preds)
    
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=['Human (Class 0)', 'AI (Class 1)']))
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(test_targets, test_preds).ravel()
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Find False Positives & False Negatives
    false_positives = []
    false_negatives = []
    
    for i, (target, pred) in enumerate(zip(test_targets, test_preds)):
        text = test_texts[i]
        if target == 0 and pred == 1:  # FP
            false_positives.append(text)
        elif target == 1 and pred == 0:  # FN
            false_negatives.append(text)
            
    # Save the model
    save_dir = "best_model"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving model and tokenizer to {save_dir}/...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save JSON metadata
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4)
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    cm = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }
    with open(os.path.join(save_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(cm, f, indent=2)
        
    stats = {
        "human_samples": 2074797,
        "ai_samples": 250001,
        "ai_models": 8,
        "domains": 4,
        "duplicates": 0,
        "missing_text": 0,
        "model_distribution": {
            "Gemini": 87591,
            "Mistral": 70948,
            "Llama": 45411,
            "Gemma": 25017,
            "Phi3": 21034
        },
        "domain_distribution": {
            "bundestag_speech": 2074797,
            "policy_document": 83333,
            "public_administration": 83334,
            "state_law": 83334
        }
    }
    with open(os.path.join(save_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    misclassifications = {
        "false_positives": false_positives[:3],  # Keep up to 3 examples
        "false_negatives": false_negatives[:3]
    }
    with open(os.path.join(save_dir, "misclassifications.json"), "w", encoding="utf-8") as f:
        json.dump(misclassifications, f, indent=2)
        
    print("Metadata JSON files successfully saved!")

if __name__ == '__main__':
    main()
