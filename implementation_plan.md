# Implementation Plan - Dataset Validation, Splitting, and Baseline Models

This plan details the implementation for validating the German AI vs. Human text dataset, resolving the group-splitting constraints, mitigating keyword leakage, and building baseline classification models (TF-IDF + Logistic Regression, and German BERT).

## User Review Required

We have identified several critical aspects of the dataset during our exploratory research that require your feedback:

> [!IMPORTANT]
> **1. The Connected Components Splitting Dilemma (Phase 2)**
> The user request states that the same `document_id` and the same `human_source_id` (speaker) must never appear in multiple splits.
> Our graph analysis shows that **95.5% of debate speakers appear in multiple documents (plenary sessions)**, and each document contains dozens of speakers. This transitively links all debate data (2,064,863 sentences, 642 documents, and 1,367 speakers) into **a single giant connected component**.
> 
> Mathematically, it is impossible to partition this debate data into Train, Validation, and Test splits without violating at least one of your two constraints. We propose three options for splitting:
> - **Option A (Recommended): Group-split by `speaker` (human_source_id) for debate data, and by `document_id` for legal data.** This guarantees no speaker leakage (preventing the classifier from memorizing speaker-specific styles), but allows sentences from the same document (plenary session) to span multiple splits.
> - **Option B: Group-split by `document_id` for all data.** This guarantees no document leakage, but allows speaker leakage (the same speaker's texts will appear in both Train and Validation/Test).
> - **Option C: Split by Connected Components.** Put 100% of debate data in Train, and partition only the legal documents across Train/Val/Test. This strictly satisfies both constraints, but leaves the Val and Test splits with 0% debate data, leading to a major domain mismatch.
> 
> *Please let us know which option you prefer.*

> [!WARNING]
> **2. Spurious Keyword Leaks (Phase 1)**
> Our frequency analysis shows that the AI dataset contains strong template-specific keyword leaks:
> - The word `plenarsitzung` appears **40,372 times in the AI sample but only 2 times in the Human sample** (Ratio = 13,457x).
> - Words like `zögert`, `hochkrempeln`, `sachorientierte`, `abwälzt`, `vorbeigehen`, `verspielt`, `drängenden`, and `nachfolgende` are highly overrepresented in AI text due to template structures used during AI text generation.
> - Date representations (like `03`, `08`, `09` in text form like "am heutigen 09.10.2025") are highly specific to AI.
> 
> If trained on raw text, any classifier will achieve ~99.9% accuracy by simply memorizing these keywords (shortcut learning) instead of learning actual *writing patterns*.
> We propose to resolve this by:
> 1. **Regex Masking**: Replacing dates with `[DATUM]`, session references with `[SITZUNG]`, and numbers with `[ZAHL]`.
> 2. **Custom Stopword Filtering**: Removing high-ratio template keywords (e.g. `plenarsitzung`, `hochkrempeln`, `zögert`, etc.) during vectorization.
> 3. **POS-Tag Features**: (Optional but recommended for TF-IDF) Vectorizing Part-of-Speech tag sequences to learn purely grammatical writing patterns.
> 
> *We will implement these preprocessing options to ensure robust learning.*

> [!NOTE]
> **3. Dataset Scale and Subsampling**
> The human dataset has 2,074,797 rows and the AI dataset has 250,001 rows (total ~2.3M rows). Training a German BERT model on 2.3M rows will take several hours, even on your RTX 4080 GPU.
> We propose to implement a `sample_size` configuration in the training scripts. This allows training and validating on a balanced, representative subset (e.g., 50,000 sentences per class) for rapid baseline iteration, while remaining fully compatible with the full dataset if you choose to run it overnight.

---

## Proposed Changes

We will create a structured python package in the workspace:

```
src/
  ├── __init__.py
  ├── utils.py            # Shared utility functions (regex masking, data loaders)
  ├── validate_dataset.py # Phase 1: Generates stats and a markdown report
  ├── create_splits.py    # Phase 2: Creates the Train/Val/Test split CSV files
  ├── train_tfidf.py      # Phase 3: Train & evaluate TF-IDF + Logistic Regression
  └── train_bert.py       # Phase 3: Fine-tune & evaluate German BERT
```

### [NEW] [utils.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/utils.py)
Contains shared code for:
- Reading the alternating AI JSONL lines correctly.
- Regex preprocessing to mask dates, session terms, numbers, and faction names.
- Defining a custom stop-word list of template leakage words.

### [NEW] [validate_dataset.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/validate_dataset.py)
Loads the full dataset and generates:
- Human vs. AI counts
- Average sentence length (characters and words) per class
- Top 100 most frequent words per class
- A detailed report saved as `artifacts/dataset_validation_report.md`.

### [NEW] [create_splits.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/create_splits.py)
Implements partitioning (Train 80%, Val 10%, Test 10%) based on the selected grouping strategy. It will:
- Partition the dataset without overlapping groups.
- Save the splits to `Human_Data/train_split.csv`, `Human_Data/val_split.csv`, and `Human_Data/test_split.csv`.
- Print verification statistics showing zero group leakage.

### [NEW] [train_tfidf.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/train_tfidf.py)
Trains the traditional ML baseline:
- Applies text normalization (masking/filtering).
- Fits a TF-IDF Vectorizer + Logistic Regression model.
- Reports Accuracy, Precision, Recall, F1, and the top predictive coefficients (to verify it did not learn leaked keywords).

### [NEW] [train_bert.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/train_bert.py)
Trains the German BERT model:
- Configurable model selection (`bert-base-german-cased` or `gbert-base`).
- Fine-tunes the model using PyTorch and HuggingFace `transformers` on the RTX 4080 GPU.
- Outputs evaluation metrics on the test split.

### [NEW] [train_stylometrics.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/train_stylometrics.py)
Trains a vocabulary-independent stylometric classifier:
- Extracts structural and grammatical features: word counts, average word lengths, Type-Token Ratio (lexical diversity), and punctuation density.
- Extracts frequencies of 15 standard German function words (stopwords) that subconscious writing style rests on, while ignoring semantic keywords.
- Trains a `RandomForestClassifier` (or `GradientBoostingClassifier`) using scikit-learn.
- Reports evaluation metrics and prints feature importances to verify what structural cues it relies on.

---

## Verification Plan

### Automated Tests
We will verify each step by running:
1. `python src/validate_dataset.py` to check that data validation statistics are computed and saved.
2. `python src/create_splits.py --strategy speaker` (or document) to create the splits and verify that no speaker or document overlaps exist between train, val, and test.
3. `python src/train_tfidf.py` to train the TF-IDF baseline and inspect the top features.
4. `python src/train_bert.py --epochs 1 --sample 50000` to verify BERT training on a sample using CUDA.
5. `python src/train_stylometrics.py` to train the Random Forest stylometric model and print its feature importances.

### Manual Verification
- We will inspect the generated `dataset_validation_report.md` to confirm counts, sentence lengths, and frequencies match the expected distributions.
- We will print the top-weighted coefficients of the Logistic Regression model to verify that none of the template-specific keywords are being used as shortcut features.
- We will inspect the feature importances of the stylometric model to ensure it is relying on grammatical features (like punctuation, lexical richness, and stopword ratios) rather than templates.
