import os
import sys
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Add parent directory to sys.path so we can import src.utils and train_stylometrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import clean_text
from src.train_stylometrics import get_stylometric_features

# Advanced text cleaner to strip out template artifacts completely
def clean_template_artifacts(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove speaker introductions
    # e.g., "Als Abgeordneter Michael Fischer der Fraktion SPD sehe ich mich in der Pflicht, ..." -> ""
    # e.g., "Im Namen der Fraktion BSW betone ich ..." -> ""
    text = re.sub(r'\bAls Abgeordneter\s+[A-ZÄÖÜa-zäöüß\-]+\s+[A-ZÄÖÜa-zäöüß\-]+\s+der Fraktion\s+[A-Z/a-zäöüß\-]+\s+(?:sehe ich mich in der Pflicht|betone ich|sehe ich)\b,?\s*(?:in dieser \d+\. Plenarsitzung)?\s*(?:beim Thema [^,]+)?\s* deutlich darauf hinzuweisen, dass\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAls Abgeordneter\s+der Fraktion\s+[A-Z/a-zäöüß\-]+\s+(?:sehe ich mich in der Pflicht|betone ich|sehe ich)\b,?\s*(?:in dieser \d+\. Plenarsitzung)?\s*(?:beim Thema [^,]+)?\s* deutliche darauf hinzuweisen, dass\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bIm Namen der Fraktion\s+[A-Z/a-zäöüß\-]+\s+betone ich\b,?\s*(?:in dieser \d+\. Plenarsitzung)?\s*(?:mit aller Deutlichkeit)?,?\s*dass\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAls Abgeordneter\s+[A-ZÄÖÜ\w\s\-]+ der Fraktion\s+\S+\b', '', text, flags=re.IGNORECASE)

    # 2. Remove Session References
    # e.g., "in dieser 137. Plenarsitzung", "in der heutigen 132. Plenarsitzung", "Plenarsitzung"
    text = re.sub(r'\b(?:in der heutigen|in dieser|in der|heutigen)?\s*\d+\.\s*(?:plenarsitzung|sitzung)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:in dieser|in der|heutigen)?\s*plenarsitzung\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bplenarsitzung\b', '', text, flags=re.IGNORECASE)
    
    # 3. Remove Aktenzeichen (Az.)
    # e.g., "unter Az. 35/7167", "Az. 19/5178"
    text = re.sub(r'\b(?:unter|gemäß|nach)?\s*Az\.\s*\d+/\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:unter|gemäß|nach)?\s*aktenzeichen\s*\d+/\d+\b', '', text, flags=re.IGNORECASE)
    
    # 4. Remove Paragraphs & Sections
    # e.g., "gemäß § 110 Abs. 3", "nach § 47 Abs. 4", "§ 110 Abs. 3", "Abs. 3"
    text = re.sub(r'\b(?:gemäß|nach|laut|nach)?\s*§+\s*\d+\s*(?:Abs\.|Absatz)\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:gemäß|nach|laut|nach)?\s*§+\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Abs\.|Absatz)\s*\d+\b', '', text, flags=re.IGNORECASE)
    
    # 5. Remove Printed Matter (Drucksache)
    # e.g., "in Drucksache 21/2312", "Drucksache 21/2312", "Drs. 21/2312"
    text = re.sub(r'\b(?:in|gemäß|nach|laut)?\s*Drucksache\s*\d+(?:/\d+)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:in|gemäß|nach|laut)?\s*Drs\.\s*\d+(?:/\d+)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDrucksache\b', '', text, flags=re.IGNORECASE)
    
    # 6. Remove Dates
    # e.g., "am heutigen 09.10.2025", "am heutigen 21.10.2019", "09.10.2025"
    text = re.sub(r'\b(?:am heutigen|heutigen)?\s*\d{2}\.\d{2}\.\d{4}\b', '', text)
    
    # 7. Clean up template phrases & keywords
    # "bezüglich [Thema]" -> "für [Thema]" or "im Bereich [Thema]"
    text = re.sub(r'\bbezüglich\b', 'für', text, flags=re.IGNORECASE)
    
    # Remove repeated template verbs or clauses
    text = re.sub(r'\bund die Lasten einseitig abwälzt', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bund die Lasten einseitig abzuwälzen', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhochkrempeln\b', 'anpacken', text, flags=re.IGNORECASE)
    text = re.sub(r'\bzögert\b', 'wartet', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsachorientierte\b', 'konstruktive', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvorbeigehen\b', 'vorübergehen', text, flags=re.IGNORECASE)
    text = re.sub(r'\bverspielt\b', 'verliert', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdrängenden\b', 'wichtigen', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnachfolgende\b', 'kommende', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgemeinwohls\b', 'Allgemeinwohls', text, flags=re.IGNORECASE)
    text = re.sub(r'\babsichtserklärungen\b', 'Erklärungen', text, flags=re.IGNORECASE)
    text = re.sub(r'\bzielvorgaben\b', 'Zielen', text, flags=re.IGNORECASE)
    text = re.sub(r'\brichtungsentscheidung\b', 'Entscheidung', text, flags=re.IGNORECASE)
    
    # 8. Clean up extra punctuation/spaces resulting from removals
    text = re.sub(r'\s+', ' ', text)  # Collapse spaces
    text = re.sub(r'\s*,\s*,', ',', text)  # Collapse double commas
    text = re.sub(r',\s*\.', '.', text)  # Clean up comma-periods
    text = re.sub(r'^\s*,\s*', '', text)  # Remove leading commas
    text = re.sub(r'\b(?:in|beim Thema|zum Bereich)\s*[\.,]', '', text, flags=re.IGNORECASE)  # Remove orphaned prepositions at end of clause
    
    # Clean up sentence starts (capitalize first letter of cleaned text)
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
        
    return text

def create_hard_set():
    print("Step 1: Creating Hard Test Set...")
    
    # 1. Load balanced test split
    test_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\test_split_balanced.csv"
    if not os.path.exists(test_path):
        print(f"Error: Balanced test split not found at {test_path}")
        return
    
    df = pd.read_csv(test_path)
    df = df.dropna(subset=['text'])
    
    # 2. Sample 1000 Human (label 0) and 1000 AI (label 1)
    df_human = df[df['label'] == 0]
    df_ai = df[df['label'] == 1]
    
    print(f"Available Human: {len(df_human)}, AI: {len(df_ai)}")
    
    df_human_sampled = df_human.sample(n=min(len(df_human), 1000), random_state=42)
    df_ai_sampled = df_ai.sample(n=min(len(df_ai), 1000), random_state=42)
    
    # 3. Clean the sampled texts
    print("Cleaning template markers in sampled texts...")
    
    df_human_sampled['cleaned_text'] = df_human_sampled['text'].apply(clean_template_artifacts)
    df_ai_sampled['cleaned_text'] = df_ai_sampled['text'].apply(clean_template_artifacts)
    
    # Create final hard test set
    hard_df = pd.concat([df_human_sampled, df_ai_sampled]).sample(frac=1.0, random_state=42)
    
    # Print some raw vs. cleaned sentences for manual inspection / logging
    print("\n--- Manual Inspection: Raw vs Cleaned Examples ---")
    ai_examples = df_ai_sampled.head(5)
    for idx, row in ai_examples.iterrows():
        print(f"\n[AI Example {idx}]")
        print(f"  Raw:     {row['text']}")
        print(f"  Cleaned: {row['cleaned_text']}")
        
    # Save the hard test set
    out_dir = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hard_test_set.csv")
    
    # We will save both the original and the cleaned version (as 'text') to make it drop-in compatible
    hard_df_save = hard_df.copy()
    # Replace the text column with the cleaned version
    hard_df_save['original_text'] = hard_df_save['text']
    hard_df_save['text'] = hard_df_save['cleaned_text']
    
    hard_df_save.to_csv(out_path, index=False)
    print(f"\nHard Test Set successfully created and saved to: {out_path}")
    
    # 4. Evaluate existing models on the Hard Test Set
    evaluate_existing_models(out_path)

def evaluate_existing_models(hard_set_path):
    print("\nEvaluating existing baseline models on the Hard Test Set...")
    
    # Load Hard Test Set
    df = pd.read_csv(hard_set_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # Train a quick TF-IDF + LR model on train_split_balanced.csv to evaluate on the Hard Test Set
    # This simulates how our existing detector performs on the Hard Test Set.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    train_path = r"c:\Users\vijayakr\Documents\Data_Analysis_Splitting\Human_Data\train_split_balanced.csv"
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path).dropna(subset=['text'])
        # Train on raw texts
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
        X_train = vectorizer.fit_transform(train_df['text'])
        y_train = train_df['label']
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        X_test = vectorizer.transform(texts)
        preds = clf.predict(X_test)
        
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        
        print("\n" + "="*50)
        print("EVALUATION: Existing TF-IDF + LR Model on Hard Test Set")
        print("="*50)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=['Human (Class 0)', 'AI (Class 1)']))
        
        # Now let's try the Stylometric RandomForest model on the Hard Test Set
        print("\nExtracting stylometric features for evaluation...")
        features_list = [get_stylometric_features(t) for t in texts]
        X_style = pd.DataFrame(features_list)
        
        # To get the training data for Random Forest
        train_features_list = [get_stylometric_features(t) for t in train_df['text']]
        X_train_style = pd.DataFrame(train_features_list)
        y_train_style = train_df['label']
        
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
        rf.fit(X_train_style, y_train_style)
        
        rf_preds = rf.predict(X_style)
        rf_acc = accuracy_score(labels, rf_preds)
        rf_f1 = f1_score(labels, rf_preds)
        
        print("\n" + "="*50)
        print("EVALUATION: Existing Stylometric RandomForest Model on Hard Test Set")
        print("="*50)
        print(f"Accuracy: {rf_acc:.4f}")
        print(f"F1-score: {rf_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, rf_preds, target_names=['Human (Class 0)', 'AI (Class 1)']))

if __name__ == '__main__':
    create_hard_set()
