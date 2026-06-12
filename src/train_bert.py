import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import clean_text

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

def load_data(filepath, use_clean=True, sample_size=None):
    print(f"Loading BERT training data from {filepath} (clean={use_clean}, sample_size={sample_size})...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text'])
    
    if sample_size is not None:
        # Balanced sampling if possible
        df_human = df[df['label'] == 0]
        df_ai = df[df['label'] == 1]
        
        n_each = sample_size // 2
        df_human_sampled = df_human.sample(n=min(len(df_human), n_each), random_state=42)
        df_ai_sampled = df_ai.sample(n=min(len(df_ai), n_each), random_state=42)
        
        df = pd.concat([df_human_sampled, df_ai_sampled]).sample(frac=1.0, random_state=42)
        
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    if use_clean:
        texts = [clean_text(t) for t in texts]
        
    return texts, labels

def main():
    parser = argparse.ArgumentParser(description="Train German BERT Classifier")
    parser.add_argument('--model_name', type=str, default='deepset/gbert-base', 
                        choices=['deepset/gbert-base', 'bert-base-german-cased'],
                        help="Pretrained HuggingFace German BERT model name")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="DataLoader batch size")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--sample_size', type=int, default=10000, 
                        help="Sample size to train on for fast baselines (balanced classes)")
    parser.add_argument('--use_raw', action='store_true', 
                        help="Use raw texts instead of cleaned/masked texts")
    
    args = parser.parse_args(args=[] if 'ipykernel' in sys.modules else None)
    
    train_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\train_split_balanced.csv"
    val_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\val_split_balanced.csv"
    test_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\test_split_balanced.csv"
    
    # 1. Load data
    use_clean = not args.use_raw
    train_texts, train_labels = load_data(train_path, use_clean=use_clean, sample_size=args.sample_size)
    val_texts, val_labels = load_data(val_path, use_clean=use_clean, sample_size=args.sample_size // 10 if args.sample_size else None)
    test_texts, test_labels = load_data(test_path, use_clean=use_clean, sample_size=args.sample_size // 10 if args.sample_size else None)
    
    # 2. Setup model and tokenizer
    print(f"Loading pretrained tokenizer and model: {args.model_name}...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # 3. Create datasets and dataloaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 5. Training Loop
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
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
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds)
        print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}")
        
    # 6. Evaluation on Test Split
    print("\nRunning evaluation on Test Split...")
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            test_preds.extend(preds)
            test_targets.extend(labels.numpy())
            
    print("\n" + "="*50)
    print(f"EVALUATION: German BERT ({args.model_name}, clean={use_clean})")
    print("="*50)
    print(f"Test Accuracy: {accuracy_score(test_targets, test_preds):.4f}")
    print(f"Test F1-score: {f1_score(test_targets, test_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=['Human (Class 0)', 'AI (Class 1)']))

if __name__ == '__main__':
    main()
