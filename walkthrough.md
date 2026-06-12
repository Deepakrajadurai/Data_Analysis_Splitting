# Walkthrough - Baseline Models and Dataset Analysis

This walkthrough details the achievements and findings for the four phases of validating the German AI vs. Human text dataset, partitioning the data safely, and training baseline and stylometric classifiers.

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

## Stylometric Classifier Results (Phase 4)

To build a model that relies exclusively on **writing patterns** rather than vocabulary, we implemented a vocabulary-independent classifier in [train_stylometrics.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/train_stylometrics.py). This extracts:
- Sentence & word lengths.
- Punctuation profiles (commas, periods, colons, parentheses, quotes).
- Nominal vs. verbal style proxy (ratio of capitalized words/nouns).
- Relative frequencies of 14 German function words (stopwords) like *der, die, das, und, ist, in, zu*.

We trained a **RandomForestClassifier** on the balanced splits:
- **Test Accuracy:** 99.82%
- **Test F1-score:** 99.83%

### Feature Importances (Top 10 Style Cues)
1. `period_count` (importance: 0.2716) — *Reflects abbreviations like "Az." and "Abs." in AI templates.*
2. `char_len` (importance: 0.1911) — *AI-generated texts are significantly longer on average.*
3. `word_len` (importance: 0.1608) — *AI texts contain more words per sentence.*
4. `comma_ratio` (importance: 0.1001) — *Reflects different subordinate clause density.*
5. `avg_word_len` (importance: 0.0552) — *German nouns/compounds are longer on average.*
6. `comma_count` (importance: 0.0363)
7. `cap_ratio` (importance: 0.0306) — *Noun vs. verb/adjective balance (nominal style).*
8. `ttr` (importance: 0.0275) — *Lexical diversity (Type-Token Ratio).*
9. `fw_die` (importance: 0.0223) — *Relative function word density.*
10. `colon_count` (importance: 0.0195)

### Verdict
Even without access to *any* semantic vocabulary words, the structural differences (sentence length, comma density, and abbreviation usage) of the template-generated AI texts are so distinct that a Random Forest classifier can distinguish them with **99.8% accuracy**. This demonstrates that AI-generated texts have a highly uniform stylistic footprint compared to natural human Bundestag speeches.
