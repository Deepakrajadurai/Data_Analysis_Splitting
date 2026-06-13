# Implementation Plan - Hard Test Set, LOMO, Adversarial Dataset, and Final Model Comparison

This plan outlines the steps to build the five chapters requested by the user: creating a hard test set, running Leave-One-Model-Out (LOMO) experiments, creating an adversarial dataset, correcting dataset distribution mismatch and leakage, and building the final comparative models.

---

## User Review Required

Please review the proposed strategies for each of the five phases.

> [!IMPORTANT]
> **1. Paraphrasing Method for Adversarial AI Dataset (Phase 3)**
> Generating 10,000 to 20,000 rewritten sentences using an LLM on the CPU can be slow (estimating 2.5–3 hours for 10k sentences using a 0.5B model).
> We propose two options to keep this process fast and efficient:
> - **Option A (Recommended)**: Use a lightweight German back-translation model (e.g., MarianMT `Helsinki-NLP/opus-mt-de-en` followed by `Helsinki-NLP/opus-mt-en-de`). Translation-back-translation is a standard, robust adversarial technique that runs at 50–100 sentences per second on CPU, taking only 3–5 minutes for 15,000 sentences.
> - **Option B**: Run a small instruction-following LLM (`Qwen/Qwen2.5-0.5B-Instruct`) on CPU on a downsampled adversarial set of 2,000 AI sentences.
> 
> *We will proceed with Option A (back-translation) for the full 15,000 sentences unless you prefer Option B.*

> [!WARNING]
> **2. Deep Learning Models Training Scale (Phase 5)**
> Training BERT and XLM-RoBERTa on CPU on the full 200,000 balanced rows is extremely computationally expensive. To run this successfully within minutes, we will downsample the improved dataset to a balanced size of **10,000 sentences** (5,000 Human, 5,000 AI) for fine-tuning BERT and XLM-RoBERTa, while evaluating the TF-IDF model on the full test split.

---

## Open Questions

- *Do you have a preference for any specific German LLM for the adversarial generation, or is our proposed back-translation / Qwen 0.5B strategy suitable?*
- *Are there any other structural keywords (beyond `Drucksache`, `Az.`, `Abs.`, session numbers, dates, and faction names) that you would like us to clean from the Hard Test Set?*

---

## Proposed Changes

We will implement the code in several new and existing files under the `src/` directory.

### [Component Name]

#### [NEW] [create_hard_test_set.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/create_hard_test_set.py)
This script will:
- Sample 1,000 human and 1,000 AI samples from the balanced test split.
- Clean them by stripping out template markers using advanced regex (case numbers `Az.`, printed papers `Drucksache`, paragraph symbols `Abs.`, date patterns, and parliamentary session references).
- Output 5 raw vs. cleaned sentences to verify the text looks natural.
- Save the hard test set to `Human_Data/hard_test_set.csv`.
- Run evaluations of the existing TF-IDF and Stylometric models on this set and report the accuracy and F1 scores.

#### [NEW] [run_lomo_experiments.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/run_lomo_experiments.py)
This script will:
- Parse the `model` or `provider` fields in `ai_generated_texts.jsonl` to group sentences by model family: Gemini, Phi3, Mistral, Gemma, and Llama.
- Run three Leave-One-Model-Out (LOMO) experiments:
  1. Train on Gemini + Phi3 + Mistral, test on Llama.
  2. Train on Gemini + Llama + Mistral, test on Phi3.
  3. Train on All except Gemma, test on Gemma.
- Evaluate a TF-IDF + Logistic Regression model on each configuration.
- Log the results in a comparative table.

#### [NEW] [create_adversarial_dataset.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/create_adversarial_dataset.py)
This script will:
- Extract 15,000 AI sentences from the train split.
- Paraphrase them using the selected method (e.g. MarianMT translation-back-translation or CPU Qwen 0.5B).
- Save them as `AI_Data/adversarial_ai_sentences.csv`.
- Evaluate the trained classifiers on these adversarial sentences to measure the performance drop (F1 drop).

#### [NEW] [improve_dataset.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/improve_dataset.py)
This script will:
- Resolve the length distribution mismatch by binning both AI and Human sentences by character length and sampling an equal number from each bin.
- Apply advanced regex masking to strip all template-based text leaks (such as `Drucksache`, `Az.`, `Abs.`, and session references) from the entire training and validation datasets.
- Save the new, leak-free, length-balanced datasets to `Human_Data/train_split_improved.csv`, `val_split_improved.csv`, and `test_split_improved.csv`.

#### [NEW] [train_final_models.py](file:///c:/Users/vijayakr/Documents/Data_Analysis_Splitting/src/train_final_models.py)
This script will:
- Train three models on the improved, leak-free, length-balanced dataset:
  1. **TF-IDF + Logistic Regression**
  2. **German BERT (`deepset/gbert-base`)**
  3. **XLM-RoBERTa (`xlm-roberta-base`)**
- Compile the evaluation metrics (Accuracy, F1-score) on the test split.
- Generate a final comparison table for the results chapter.

---

## Verification Plan

### Automated Tests
1. `python src/create_hard_test_set.py` to create `hard_test_set.csv` and evaluate baseline models on it.
2. `python src/run_lomo_experiments.py` to run LOMO experiments and generate the F1 results table.
3. `python src/create_adversarial_dataset.py` to generate the adversarial dataset and evaluate the detectors.
4. `python src/improve_dataset.py` to produce the length-balanced, leak-free training splits.
5. `python src/train_final_models.py` to run the comparative final model training and print the metrics table.

### Manual Verification
- We will visually inspect the raw vs. cleaned sentences printed by `create_hard_test_set.py` to make sure they look clean and natural.
- We will verify the length histograms of the improved dataset to confirm that the distributions match.
- We will inspect the F1 metrics across unseen models and adversarial sets to draw scientific conclusions for the thesis.
