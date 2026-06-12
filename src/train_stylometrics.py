import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# German function words (stopwords) that reflect syntactic connection habits rather than content
FUNCTION_WORDS = ['der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'den', 'von', 'mit', 'dass', 'sich', 'auf', 'für']

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_stylometric_features(text):
    features = {}
    
    # 1. Text & word length metrics
    char_len = len(text)
    orig_tokens = re.findall(r'\b\w+\b', text)
    word_len = len(orig_tokens)
    
    features['char_len'] = char_len
    features['word_len'] = word_len
    features['avg_word_len'] = char_len / word_len if word_len > 0 else 0
    
    # 2. Lexical diversity (Type-Token Ratio)
    tokens_lower = [t.lower() for t in orig_tokens]
    unique_words = len(set(tokens_lower))
    features['ttr'] = unique_words / word_len if word_len > 0 else 0
    
    # 3. Capitalization habits (nominal vs verbal style proxy)
    # German nouns are capitalized. Higher cap_ratio indicates nominal style.
    cap_count = sum(1 for t in orig_tokens if t and t[0].isupper())
    features['cap_ratio'] = cap_count / word_len if word_len > 0 else 0
    
    # 4. Punctuation counts & ratios (captures syntactic complexity)
    comma_count = text.count(',')
    features['comma_count'] = comma_count
    features['period_count'] = text.count('.')
    features['colon_count'] = text.count(':')
    features['semicolon_count'] = text.count(';')
    features['qmark_count'] = text.count('?')
    features['excl_count'] = text.count('!')
    features['paren_count'] = text.count('(') + text.count(')')
    features['quote_count'] = text.count('"') + text.count("'") + text.count('„') + text.count('“')
    
    features['comma_ratio'] = comma_count / word_len if word_len > 0 else 0
    
    # 5. Function word distributions (relative frequencies)
    word_counts = {}
    for w in tokens_lower:
        word_counts[w] = word_counts.get(w, 0) + 1
        
    for fw in FUNCTION_WORDS:
        features[f'fw_{fw}'] = word_counts.get(fw, 0) / word_len if word_len > 0 else 0
        
    return features

def load_stylometric_dataset(filepath):
    print(f"Loading and extracting stylometrics from {filepath}...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text'])
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    features_list = []
    for t in tqdm(texts, desc="Extracting features"):
        features_list.append(get_stylometric_features(t))
        
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    return X, y

def main():
    train_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\train_split_balanced.csv"
    test_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\test_split_balanced.csv"
    
    # Load and extract features
    X_train, y_train = load_stylometric_dataset(train_path)
    X_test, y_test = load_stylometric_dataset(test_path)
    
    print("\nTraining Random Forest Stylometric Classifier...")
    # Using RandomForestClassifier with fast parameters
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    clf.fit(X_train, y_train)
    
    # Evaluate
    preds = clf.predict(X_test)
    
    print("\n" + "="*50)
    print("EVALUATION: RandomForest Stylometric Classifier")
    print("="*50)
    print(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Test F1-score: {f1_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Human (Class 0)', 'AI (Class 1)']))
    
    # Inspect feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns
    
    print("\nFeature Importances (Top 20 Stylometric Markers):")
    for rank in range(min(20, len(indices))):
        idx = indices[rank]
        print(f"  {rank+1:2d}. {feature_names[idx]:20s} (importance: {importances[idx]:.4f})")

if __name__ == '__main__':
    main()
