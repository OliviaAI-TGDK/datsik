"""
TGDK Dataset Preparer — Datsik Edition
--------------------------------------
Loads JSONL/CSV/TXT training data from /packs,
handles Windows smart-quotes and non-UTF-8 encodings,
and saves tokenized output for QLoRA fine-tuning.
"""

import os, glob, pyarrow as pa
import pyarrow.csv as csv
import pyarrow.json as pj
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKS_DIR = os.path.join(BASE_DIR, "packs")
OUT_DIR = os.path.join(PACKS_DIR, "tokenized")

os.makedirs(OUT_DIR, exist_ok=True)

train_files = glob.glob(os.path.join(PACKS_DIR, "train*.jsonl"))
val_files   = glob.glob(os.path.join(PACKS_DIR, "val*.jsonl"))

if not train_files:
    raise FileNotFoundError(f"No training files found in {PACKS_DIR}")
if not val_files:
    print("⚠️ No validation files found, using train set for validation.")
    val_files = train_files

print(f"📂 Found {len(train_files)} train and {len(val_files)} val files in /packs")

# --- Force Latin-1 decode at load time ---
def load_text_dataset(paths):
    merged_tables = []
    for path in paths:
        try:
            table = pj.read_json(
                path,
                read_options=pj.ReadOptions(block_size=1 << 20)
            )
        except Exception:
            table = csv.read_csv(
                path,
                read_options=csv.ReadOptions(encoding="latin1")
            )
        merged_tables.append(table)
    table = pa.concat_tables(merged_tables)
    df = table.to_pandas().astype(str)
    return Dataset.from_pandas(df)

# === Load datasets ===
print("📦 Loading datasets...")
try:
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_files,
            "validation": val_files
        },
        encoding="latin-1",
        keep_in_memory=True
    )
except Exception as e:
    print(f"⚠️ load_dataset failed ({e}), falling back to Arrow read...")
    dataset = {
        "train": load_text_dataset(train_files),
        "validation": load_text_dataset(val_files)
    }

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1", use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

# === Text formatter ===
def format_text(example):
    if "text" in example:
        return example["text"]
    elif "instruction" in example and "output" in example:
        return f"Instruction: {example['instruction']}\nResponse: {example['output']}"
    elif "prompt" in example and "completion" in example:
        return f"{example['prompt']}\n{example['completion']}"
    else:
        return str(example)

# === Safe decode ===
def safe_decode(t):
    if isinstance(t, bytes):
        try:
            return t.decode("utf-8")
        except UnicodeDecodeError:
            return t.decode("latin-1", errors="replace")
    elif isinstance(t, str):
        try:
            t.encode("utf-8")
            return t
        except UnicodeEncodeError:
            return t.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
    return str(t)

# === Tokenization ===
def tokenize_fn(batch):
    texts = [safe_decode(format_text(e)) for e in batch["text"]] if "text" in batch else [
        safe_decode(format_text(e)) for e in batch.values()
    ]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors=None,
    )

print("🧠 Tokenizing...")
if isinstance(dataset, dict):  # fallback dict mode
    tokenized_train = dataset["train"].map(tokenize_fn, batched=True)
    tokenized_val = dataset["validation"].map(tokenize_fn, batched=True)
else:
    tokenized_train = dataset["train"].map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_val = dataset["validation"].map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

# === Save ===
print("💾 Saving tokenized datasets...")
tokenized_train.save_to_disk(os.path.join(OUT_DIR, "train"))
tokenized_val.save_to_disk(os.path.join(OUT_DIR, "val"))
print(f"✅ Tokenized dataset saved at: {OUT_DIR}")
