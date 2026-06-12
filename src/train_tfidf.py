import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import clean_text, TEMPLATE_KEYWORDS

def load_data(filepath, use_clean=True):
    print(f"Loading data from {filepath} (clean={use_clean})...")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['text'])
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    if use_clean:
        # Apply regex cleaning to each text
        texts = [clean_text(t) for t in texts]
        
    return texts, labels

def train_and_eval(train_path, test_path, use_clean=True):
    # 1. Load splits
    train_texts, train_labels = load_data(train_path, use_clean=use_clean)
    test_texts, test_labels = load_data(test_path, use_clean=use_clean)
    
    # 2. Vectorize using TF-IDF (1-grams and 2-grams)
    # If using clean, we also exclude the custom TEMPLATE_KEYWORDS stopwords.
    stop_words = list(TEMPLATE_KEYWORDS) if use_clean else None
    
    print("Vectorizing texts with TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        stop_words=stop_words
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    # 3. Fit Logistic Regression
    print("Fitting Logistic Regression baseline...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train, train_labels)
    
    # 4. Predict and Evaluate
    preds = clf.predict(X_test)
    
    print("\n" + "="*50)
    print(f"EVALUATION: TF-IDF + Logistic Regression (clean={use_clean})")
    print("="*50)
    print(f"Test Accuracy: {accuracy_score(test_labels, preds):.4f}")
    print(f"Test F1-score: {f1_score(test_labels, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=['Human (Class 0)', 'AI (Class 1)']))
    
    # 5. Extract top features to inspect shortcut learning
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]
    
    top_ai_indices = np.argsort(coef)[-20:][::-1]
    top_human_indices = np.argsort(coef)[:20]
    
    print("\nTop 20 most predictive features for AI (Class 1):")
    for i, idx in enumerate(top_ai_indices):
        print(f"  {i+1:2d}. {feature_names[idx]:30s} (coeff: {coef[idx]:.4f})")
        
    print("\nTop 20 most predictive features for Human (Class 0):")
    for i, idx in enumerate(top_human_indices):
        print(f"  {i+1:2d}. {feature_names[idx]:30s} (coeff: {coef[idx]:.4f})")
        
    return accuracy_score(test_labels, preds), f1_score(test_labels, preds)

def main():
    train_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\train_split_balanced.csv"
    test_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\test_split_balanced.csv"
    
    # Compare raw vs cleaned texts
    print("--- Configuration 1: Raw Texts (Subject to Shortcut Learning) ---")
    train_and_eval(train_path, test_path, use_clean=False)
    
    print("\n" + "#"*80 + "\n")
    
    print("--- Configuration 2: Cleaned & Masked Texts (Robust Writing Patterns) ---")
    train_and_eval(train_path, test_path, use_clean=True)

if __name__ == '__main__':
    main()
