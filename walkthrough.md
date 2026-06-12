# Walkthrough - Baseline Models and Dataset Analysis

This walkthrough details the achievements and findings for the three phases of validating the German AI vs. Human text dataset, partitioning the data safely, and training baseline classifiers.

---

## What Was Checked & Validated (Phase 1)

We scanned the entire dataset (2,074,797 human sentences, 250,001 AI sentences) to generate the statistics now saved in [dataset_validation_report.md](file:///C:/Users/vijayakr/.gemini/antigravity-ide/brain/4f030afc-573a-438e-b7fb-239b35d0d759/dataset_validation_report.md).

### 1. Class Counts & Length Distributions
- **Human Count:** 2,074,797 sentences
- **AI Count:** 250,001 sentences
- **Average Sentence Length:**
  - **Human:** 118.78 characters (17.14 words)
  - **AI:** 195.31 characters (25.83 words) *(AI sentences are 64.4% longer on average)*

### 2. Spurious Keyword Leaks (Shortcut Learning)
We discovered severe vocabulary leakage in the AI dataset. AI generation scripts used highly rigid sentence templates that led to the extreme overrepresentation of certain words:
- `plenarsitzung` appears **40,372 times in AI** but only **2 times in Human** (Ratio = 20,186x).
- `zögert` appears **8,365 times in AI** but **0 times in Human** (Ratio = $\infty$).
- Date markers (e.g. `09.10.2025`, `14.06.2024`) and template verbs (`hochkrempeln`, `abwälzt`, `vorbeigehen`) act as direct "shortcuts" for a classifier.

---

## Data Splitting Verification (Phase 2)

We successfully partitioned the dataset into Train (80%), Validation (10%), and Test (10%) splits.
To resolve the **bipartite connected component** issue (where 95.5% of speakers are linked across multiple plenary sessions), we implemented **Option A**:
- **Debate data** was grouped and split by `speaker` to prevent speaker memorization.
- **Legal data** was grouped and split by `document_id` (e.g., `zpo`, `bgb`).
- **AI data** was split by unique document IDs.

We saved both the full splits and balanced downsampled splits (`train_split_balanced.csv` containing 100k human and 100k AI rows) in `Human_Data/`.

### Split Group Leakage Check
The split script verified that there is **absolute zero group leakage** across the splits:
- `Overlap Train-Val Groups`: 0
- `Overlap Train-Test Groups`: 0
- `Overlap Val-Test Groups`: 0

---

## Baseline Model Evaluations (Phase 3)

### Model 1: TF-IDF + Logistic Regression
We trained the model in two configurations on the balanced splits (200k train, 20k test) to evaluate the impact of keyword cleaning:

#### Configuration 1: Raw Text
- **Test Accuracy / F1:** 99.99%
- **Top AI Features:** `bezüglich` (coef: 13.64), `abs` (coef: 9.22), `unter` (coef: 8.95), `az` (coef: 8.69), `drucksache` (coef: 8.30), `plenarsitzung` (coef: 6.48), and date markers.
- **Verdict:** The classifier achieved perfect accuracy by simply memorizing the template keywords rather than learning style patterns.

#### Configuration 2: Cleaned & Masked Text (with Custom Stopwords)
- **Test Accuracy / F1:** 100.00%
- **Top AI Features:** `im` (coef: 8.36), `zahl` (coef: 8.23), `drucksache` (coef: 8.03), `unter` (coef: 8.00), `az` (coef: 7.50), `abs` (coef: 7.08).
- **Verdict:** Even after masking dates/numbers/parties and removing the highest-ratio words, the model still achieved 100.00% accuracy by latching onto other structural template words (like `az`, `abs`, `drucksache`, `heutigen`, `thema`).

---

### Model 2: German BERT (`deepset/gbert-base`)
Due to Python 3.14 compatibility limitations on Windows (which lack precompiled PyTorch CUDA wheels), we successfully installed `sentencepiece` and ran fine-tuning on a balanced sample of **2,000 sentences for 1 epoch on the CPU**.
- **Epoch 1 Train Loss:** 0.1007
- **Validation Accuracy:** 100.00%
- **Test Accuracy / F1:** 100.00% (on the 200 test set sentences)
- **Verdict:** BERT easily achieved 100% accuracy on the cleaned/masked data, confirming that it also exploits the remaining structural template patterns in the AI-generated texts.

---

## Key Recommendations & Next Steps

1. **Dataset Bias Mitigation:** The 100% accuracy of both baseline models on cleaned data shows that the AI texts have a highly uniform template structure. The model is classifying *sentence templates* rather than *writing style*.
2. **Further Structural Masking:** To build a model that generalizes to natural human and AI writing, we recommend masking structural terms like `az` (case number), `abs` (paragraph), and `drucksache` (parliamentary print).
3. **Punctuation and Grammatical Training:** We suggest training a classifier purely on **POS (Part-of-Speech) tag sequences** (e.g. `PRON VERB DET NOUN PUNCT`) to force the models to focus exclusively on grammatical and sentence-structure distributions.
