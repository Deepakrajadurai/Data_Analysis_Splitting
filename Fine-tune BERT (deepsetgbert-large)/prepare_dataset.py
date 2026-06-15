"""
Step 1 — Dataset Preparation
==============================
- Samples 500K human (label=0) + 250K AI (label=1) from your full CSVs
- Cleans and validates text quality
- Stratified 80 / 10 / 10 train/val/test split
- Saves splits to data/train.csv, data/val.csv, data/test.csv

Usage:
    python 01_prepare_dataset.py \
        --human_csv  data/bundestag_sentences.csv \
        --ai_csv     data/ai_generated_sentences.csv
"""

import argparse
import logging
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HUMAN_SAMPLE    = 500_000
AI_SAMPLE       = 250_000
MIN_WORDS       = 20
MAX_CHARS       = 1_024      # BERT max token safety: ~512 tokens ≈ 1024 chars
RANDOM_SEED     = 42
OUTPUT_DIR      = Path("data")


# ---------------------------------------------------------------------------
# CLEANING
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str | None:
    """Return cleaned text or None if it fails quality checks."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Word count
    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    # Char length cap (keeps within BERT limits)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0]  # truncate at word boundary
    # Reject if >30% digits (likely a table / numerical artefact)
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.30:
        return None
    # Rough German check: must contain at least one German-only character
    if not re.search(r"[äöüÄÖÜß]", text):
        return None
    return text


def load_and_clean(csv_path: Path, label: int, n_sample: int) -> pd.DataFrame:
    log.info(f"Loading {csv_path} (label={label}) ...")
    df = pd.read_csv(csv_path, usecols=["text", "label",
                                         "source", "wahlperiode",
                                         "datum",  "speaker"],
                     dtype=str, low_memory=False)
    df["label"] = label   # enforce correct label

    log.info(f"  Raw rows: {len(df):,}")
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    log.info(f"  After cleaning: {len(df):,}")

    if len(df) < n_sample:
        log.warning(f"  Only {len(df):,} rows available, wanted {n_sample:,}. Using all.")
    else:
        df = df.sample(n=n_sample, random_state=RANDOM_SEED)

    log.info(f"  Sampled: {len(df):,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(human_csv: str, ai_csv: str):
    OUTPUT_DIR.mkdir(exist_ok=True)

    human_df = load_and_clean(Path(human_csv), label=0, n_sample=HUMAN_SAMPLE)
    ai_df    = load_and_clean(Path(ai_csv),    label=1, n_sample=AI_SAMPLE)

    df = pd.concat([human_df, ai_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    log.info(f"\nCombined dataset: {len(df):,} rows")
    log.info(f"Label distribution:\n{df['label'].value_counts()}")

    # ---- Stratified 80 / 10 / 10 split ------------------------------------
    train_df, temp_df = train_test_split(df, test_size=0.20,
                                          stratify=df["label"],
                                          random_state=RANDOM_SEED)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50,
                                          stratify=temp_df["label"],
                                          random_state=RANDOM_SEED)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR  / "val.csv",   index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv",  index=False)

    log.info(f"\nSplits saved:")
    log.info(f"  train : {len(train_df):,} rows → data/train.csv")
    log.info(f"  val   : {len(val_df):,}   rows → data/val.csv")
    log.info(f"  test  : {len(test_df):,}  rows → data/test.csv")

    # Sanity check label balance per split
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split["label"].value_counts()
        log.info(f"  {name} label dist: human={counts.get(0,0):,}  ai={counts.get(1,0):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_csv", default="data/bundestag_sentences.csv")
    parser.add_argument("--ai_csv",    default="data/ai_generated_sentences.csv")
    args = parser.parse_args()
    main(args.human_csv, args.ai_csv)