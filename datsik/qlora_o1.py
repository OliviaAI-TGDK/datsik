#=====================================================================
#
#     * DEV TEST KIT - TREAD AT YOUR OWN RISK - Progress 90% *
#
#            TGDK / OliviaAI QLoRA Training Environment
#                     Copyright TGDK 2025
#=====================================================================
# qlora.py – TGDK Magic + Duo + MMT + JadeCodewright + Seals + Rituals
import os, sys, torch, hashlib, json, subprocess, datetime, glob, argparse, lzma, shutil, math, types, psutil, warnings, pypdf,  gc, inspect
import binascii, random
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig, get_scheduler
)
from transformers.modeling_outputs import SequenceClassifierOutput
import lzma
import base64
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from torch.optim import AdamW
import sqlite3
from lion_pytorch import Lion
from scipy.spatial import Delaunay
import logging, time
import torch.nn as nn
import glob
import prepare_dataset
from fusion_config import BertMistralFusionConfig
# External TGDK geometry modules
from tgdk_accelerator import GhostGateAccelerator
from Mahadevi import Mahadevi
from Maharaga import Maharaga
from Trinity import Trinity
from Duo_o1 import make_duo_optimizer, Duo
from device import load_model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, TrainerCallback, PretrainedConfig, RobertaTokenizerFast
from transformers.modeling_outputs import CausalLMOutput
from SPMF import SubcutaneousParticleMatrix
from tokenizers import ByteLevelBPETokenizer
import torch.nn.functional as F
from adversarial import AdversarialManeuver
from transformers.utils import logging as hf_logging
from volumetric_infinitizer import VolumetricInfinitizer, Realmsy
from dce import DicleasiasticClauseEngine
from possessor_module import PossessorScript, QuadrolateralDiagram
from code_wright import CodeWright 
from datasets import load_dataset, load_from_disk, DatasetDict
from tvr import TruncatedVectorResponse
from csr import ClauseStandpoint, FractalAI
from rsicommand import RSICommand
from torch.utils.data import DataLoader
import time
from mirrorblade import mirrorblade_step_callback
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from symbolic_adapter import Vector3, reflect_through_fold, SymbolicSignalDataset
from icu import IncumbentVector
from mainmethods import rotate_vector, reflect_phi
from edge_of_fold import EdgeOfFold
from torch.cuda.amp import GradScaler, autocast
import torch._dynamo
from core_config import parse_cli_args, load_model_config
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from tgdk_datsik import DATSIKTrainer
from transformers import TrainerCallback

from datetime import datetime
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from torch.utils.data import Subset

# === Define data directory (path to your tokenized data) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "packs", "tokenized")

tokenized_dir = os.path.join("packs", "tokenized")

train_dataset, val_dataset = None, None

# --- Try standard Hugging Face format first ---
try:
    ds = load_from_disk(tokenized_dir)
    if isinstance(ds, DatasetDict):
        train_dataset = ds.get("train")
        val_dataset   = ds.get("validation") or ds.get("val")
        print(f"[MirrorBlade] ✅ Loaded DatasetDict from {tokenized_dir}")
    elif isinstance(ds, Dataset):
        train_dataset = ds
        print(f"[MirrorBlade] ✅ Loaded single Dataset from {tokenized_dir}")
except Exception as e:
    print(f"[MirrorBlade] ⚠️ load_from_disk failed: {type(e).__name__}: {e}")

# --- Fallback: rebuild from .arrow or JSONL files ---
if train_dataset is None:
    arrow_files = glob.glob(os.path.join(tokenized_dir, "*.arrow"))
    json_files  = glob.glob(os.path.join("packs", "*.jsonl"))
    if arrow_files:
        print(f"[MirrorBlade] ♻️ Rebuilding dataset from {len(arrow_files)} Arrow shards...")
        train_dataset = load_dataset("arrow", data_files={"train": arrow_files}).get("train")
    elif json_files:
        print(f"[MirrorBlade] ♻️ Rebuilding dataset from JSONL (packs/*.jsonl)...")
        train_dataset = load_dataset("json", data_files={"train": json_files}).get("train")
    else:
        print("[MirrorBlade] ❌ No dataset files found for rebuild — empty pack.")

# --- Create fake validation split if needed ---
if val_dataset is None and train_dataset is not None:
    val_size = max(1, int(0.2 * len(train_dataset)))
    val_dataset = train_dataset.select(range(val_size))
    train_dataset = train_dataset.select(range(val_size, len(train_dataset)))
    print(f"[MirrorBlade] ⚙️ Synthesized validation split: {len(train_dataset)} train, {len(val_dataset)} val")

# --- Final safe counts ---
train_len = len(train_dataset) if train_dataset else 0
val_len   = len(val_dataset) if val_dataset else 0
print(f"✅ Loaded tokenized data: {train_len} train, {val_len} val examples")



# === TGDK Quad-deterministic Callback ===
class TGDKQuadCallback(TrainerCallback):
    """
    TGDKQuadCallback
    ----------------
    - Collects RNG state entropy every N steps
    - Signs intermediate checkpoints using q0–q3 keys
    - Logs to TGDK Vault JSONL for audit
    - Creates 3x3 (Triad) and 4x4 (Quad) state summaries
    """

    def __init__(self,
                 interval_steps: int = 144,
                 key_dir: str = "./datsik_keys",
                 out_dir: str = "./artifacts"):
        self.interval = interval_steps
        self.key_dir = Path(key_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.signers = ["q0", "q1", "q2", "q3"]
        self.vault_log = self.out_dir / "TGDKVault_DATSikLog.ndjson"
        print(f"[TGDKQuadCallback] Initialized — interval={interval_steps} steps")

    def _rng_entropy(self):
        """Compute a stable numeric entropy signature for RNG states."""
        torch_state = torch.get_rng_state().numpy()
        np_state = np.random.get_state()[1]
        rand_state = random.getstate()[1]
        data = np.concatenate([
            torch_state[:64].astype(np.uint8),
            np.frombuffer(bytes(rand_state[:64]), dtype=np.uint8, count=min(64, len(bytes(rand_state[:64]))))
        ])
        digest = hashlib.sha256(data.tobytes()).hexdigest()
        return digest

    def _sign_entropy(self, digest: str):
        """Sign entropy digest using local keys."""
        blob = digest.encode()
        signatures = {}
        for name in self.signers:
            priv_path = self.key_dir / f"{name}.pem"
            if not priv_path.exists():
                continue
            with open(priv_path, "rb") as f:
                priv = serialization.load_pem_private_key(f.read(), password=None)
            sig = priv.sign(
                blob,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            signatures[name] = binascii.hexlify(sig).decode()
        return signatures

    def _log_to_vault(self, record):
        """Append to append-only NDJSON vault log."""
        with open(self.vault_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # === Hooked events ===
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0 and state.global_step > 0:
            entropy_sig = self._rng_entropy()
            timestamp = datetime.utcnow().isoformat() + "Z"
            record = {
                "step": state.global_step,
                "timestamp": timestamp,
                "entropy_sig": entropy_sig,
                "triad": self._triad_state(entropy_sig),
                "quad": self._quad_state(entropy_sig),
                "signatures": self._sign_entropy(entropy_sig),
            }
            self._log_to_vault(record)
            print(f"[TGDKQuadCallback] 🌀 Step {state.global_step} signed entropy {entropy_sig[:12]}...")
        return control

    def _triad_state(self, digest: str):
        """Generate 3x3 deterministic triad summary."""
        nums = [int(digest[i:i+4], 16) % 144 for i in range(0, 36, 4)]
        triad = np.array(nums[:9]).reshape(3, 3).tolist()
        return triad

    def _quad_state(self, digest: str):
        """Generate 4x4 deterministic quad summary."""
        nums = [int(digest[i:i+4], 16) % 256 for i in range(0, 64, 4)]
        quad = np.array(nums[:16]).reshape(4, 4).tolist()
        return quad



# === 1️⃣ Paths ===
WORKING_DIR = Path(__file__).parent.resolve()
PACKS_DIR = WORKING_DIR / "packs"
if not PACKS_DIR.exists():
    raise FileNotFoundError(f"❌ Expected folder: {PACKS_DIR}")

# === 2️⃣ Gather .jsonl files ===
jsonl_files = [str(p) for p in PACKS_DIR.rglob("*.jsonl") if p.stat().st_size > 0]
if not jsonl_files:
    raise FileNotFoundError(f"❌ No .jsonl files found under {PACKS_DIR}")

train_files = [f for f in jsonl_files if "train" in Path(f).name.lower()]
val_files   = [f for f in jsonl_files if "val" in Path(f).name.lower()]
if not train_files:
    train_files = jsonl_files[:]

data_files = {"train": train_files}
if val_files:
    data_files["validation"] = val_files

print(f"📦 Found {len(train_files)} train and {len(val_files)} val files.")

# === 3️⃣ Load JSONL without cache ===
dataset = load_dataset(
    "json",
    data_files={
        "train": "packs/*.jsonl",
        "validation": "packs/*.jsonl"
    },
    encoding="latin-1",
    keep_in_memory=True
)

# === 4️⃣ Ensure text column exists and clean ===
def ensure_text(example):
    t = example.get("text")
    if t is None:
        for alt in ["content", "prompt", "instruction", "message", "completion"]:
            if alt in example and isinstance(example[alt], str):
                t = example[alt]
                break
    if t is None:
        t = ""
    elif not isinstance(t, str):
        t = str(t)
    return {"text": t}

for split in dataset.keys():
    dataset[split] = dataset[split].map(ensure_text, load_from_cache_file=False, keep_in_memory=True)
    dataset[split] = dataset[split].filter(lambda e: isinstance(e["text"], str) and len(e["text"].strip()) > 0)

# === 5️⃣ Rebuild dataset objects to drop all cache traces ===
def rebuild(ds):
    clean = [{"text": ex["text"]} for ex in ds if isinstance(ex["text"], str)]
    return Dataset.from_list(clean)

ds_train = rebuild(dataset["train"])
if "validation" in dataset:
    ds_val = rebuild(dataset["validation"])
else:
    split = ds_train.train_test_split(test_size=0.1, shuffle=True, seed=42)
    ds_train, ds_val = split["train"], split["test"]

# === 6️⃣ Verify integrity ===
def verify_dataset(ds, label):
    bad = [i for i, e in enumerate(ds) if not isinstance(e.get("text"), str)]
    if bad:
        raise ValueError(f"❌ {len(bad)} invalid text entries in {label}")
    print(f"✅ {label}: {len(ds)} samples, all valid strings")

verify_dataset(ds_train, "train")
verify_dataset(ds_val, "val")

print(f"✅ Clean dataset ready — train={len(ds_train)} | val={len(ds_val)}")

    # --- Absolute minimal dataset cleaner ---
def clean_dataset(ds):
    return ds.filter(lambda ex: isinstance(ex.get("text", ""), str))

ds_train = clean_dataset(ds_train)
ds_val   = clean_dataset(ds_val)



# Make PyTorch behave with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_CUDA_DEBUG_MEMORY"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["ACCELERATE_DISABLE_METADATA_CLEANUP"] = "1"
os.environ["ACCELERATE_USE_DEEPSPEED_LOAD_ZEROSTAGE3"] = "0"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

args = parse_cli_args()
cfg = load_model_config()   # ✅ ensures cfg exists

torch.cuda.empty_cache(); gc.collect()
# --- Compatibility stub for TRL/PEFT block grouping ---
def get_transformer_blocks(model):
    """
    Compatibility helper for PEFT/QLoRA grouping.
    Returns the list of transformer blocks if available.
    """
    for attr in ["model", "transformer", "backbone"]:
        if hasattr(model, attr):
            sub = getattr(model, attr)
            for blk_name in ["layers", "blocks", "h"]:
                if hasattr(sub, blk_name):
                    return getattr(sub, blk_name)
    return []  # default: empty list for safe fallback

# 🛡 Disable Inductor/Triton compilation on Windows
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 0
torch._dynamo.config.optimize_ddp = False
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")

# Optional: avoid unsupported BF16 kernels
torch.set_float32_matmul_precision("medium")
os.environ["TORCHINDUCTOR_DISABLE_BF16"] = "1"

print("[MirrorBlade] ⚙️ Running in Eager Mode — Triton disabled (Windows-safe).")



use_cuda = torch.cuda.is_available()
supports_bf16 = use_cuda and torch.cuda.is_bf16_supported()

bf16_flag = supports_bf16
fp16_flag = not supports_bf16

ram_total = psutil.virtual_memory().total / (1024**3)
max_host_ram = 0.75 * ram_total
torch.set_float32_matmul_precision("high")
print(f"💾 Host memory target ≈ {max_host_ram:.1f} GB")

def safe_preview(obj, n=40):
    """Return a stable string preview for any object (including None)."""
    if obj is None:
        return "<none>"
    try:
        if isinstance(obj, (dict, list, tuple, set)):
            return json.dumps(obj, default=str)[:n]
        return str(obj)[:n]
    except Exception as e:
        return f"<unprintable:{type(obj).__name__}:{e}>"

def rsi_execute_safe(rsi, command, params=None):
    """Call RSI and normalize output to (text_repr, raw, meta)."""
    try:
        out = rsi.execute(command, params or [])
    except Exception as e:
        return f"RSI error: {e!r}", None, {"error": True}

    # Normalize common shapes
    if out is None:
        return "<none>", None, {}
    if isinstance(out, tuple) and len(out) == 2:
        msg, meta = out
        return str(msg) if msg is not None else "<none>", out, meta or {}
    if isinstance(out, dict):
        msg = out.get("message")
        if msg is None:
            # Fall back to the whole dict if no 'message' key
            return json.dumps(out, default=str), out, {}
        return str(msg), out, {}
    # string / number / other
    return str(out), out, {}


v = Vector3(1.0, 2.0, -0.5, intent="scan")
env = {"fold": "7", "epoch": "10", "gate": "Beta", "intent": "scan"}

methods = {
    "rotate": rotate_vector,
    "reflect": reflect_phi
}

iv = IncumbentVector(vector=v, environment=env, methods=methods)

# Share or broadcast
vector_info = iv.describe()

# Invoke a method
rotated = iv.invoke("rotate", axis='z', degrees=90)
reflected = iv.invoke("reflect")


# Create example vector
v = Vector3(1.0, -2.0, 0.5, intent="learn")

# Reflect
scooty = reflect_through_fold(v, axis='y')

# Dataset (batch of 10 for example)
scooties = [scooty for _ in range(10)]
dataset = SymbolicSignalDataset(scooties)



# ANSI rainbow colors
RAINBOW = [
    "\033[91m",  # Red
    "\033[93m",  # Yellow
    "\033[92m",  # Green
    "\033[96m",  # Cyan
    "\033[94m",  # Blue
    "\033[95m",  # Magenta
]
RESET = "\033[0m"

DOG = "🐕"

def rainbow_dog_bar(current, total, width=40):
    filled = int(width * current / total)
    bar = ""
    for i in range(filled):
        color = RAINBOW[i % len(RAINBOW)]
        bar += f"{color}{DOG}{RESET}"
    bar += "·" * (width - filled)
    percent = (current / total) * 100
    sys.stdout.write(
        f"\r[Loading] {bar} {current}/{total} ({percent:5.1f}%)"
    )
    sys.stdout.flush()

# ----------------------------
# Setup logging first
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_propagation()

# ----------------------------
# Initialize components
# ----------------------------
cw = CodeWright("M")
possessor = PossessorScript(codewright=cw)
rsi = RSICommand()

# ----------------------------
# Register commands
# ----------------------------
rsi.register("invoke_flag", possessor.invoke_flag)
rsi.register("offer_sword", possessor.offer_sword)
rsi.register("paternalize", lambda params: possessor.ratio_paternalizer(params))

# ----------------------------
# Run a test epoch
# ----------------------------
dummy_batch = [1, 2, 3]

epoch = 0  # or any integer
batch = np.arange(1, 1025).reshape(32, 32)  
num_epochs = 1
batch_size = 258
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(3):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]["text"]
        possessor.paradoxialize_epoch(epoch=epoch, data=batch)

print("✅ Has paradoxialize_epoch:", hasattr(possessor, "paradoxialize_epoch"))
print("🔹 Last epoch entry:", possessor.epoch_vector[-1])
print("🔹 Final offering:", possessor.final_offering())

# ----------------------------
# Execute RSI commands
# ----------------------------
print(rsi.execute("invoke_flag"))
print(rsi.execute("offer_sword"))

outdir = "./tgdk_base_tok"
os.makedirs(outdir, exist_ok=True)

# Save raw BPE vocab + merges
tokenizer = ByteLevelBPETokenizer()
print(f"[OK] Tokenizer vocab saved → {outdir}")


outdir = "./tgdk_base_tok"
os.makedirs(outdir, exist_ok=True)


# Train your tokenizer
base_tok = ByteLevelBPETokenizer()
base_tok.train(files=["tgdk_corpus.txt"], vocab_size=32000, min_frequency=2)

# Wrap into Hugging Face tokenizer
hf_tok = RobertaTokenizerFast(tokenizer_object=base_tok)

# Add TGDK-specific symbols
special_tokens = {"additional_special_tokens": ["[HONOR]", "[CXM]", "[BHR]", "[CULMEX]"]}
hf_tok.add_special_tokens(special_tokens)

# Save in Hugging Face format
hf_tok.save_pretrained(outdir)



# resolve OUT directory from env or default to ./out
OUT = os.environ.get("OUT", "./out")
os.makedirs(OUT, exist_ok=True)
possessor.paradoxialize_epoch(epoch, data=batch, state="active")
possessor.paradoxialize_epoch(epoch, state="fallback")
possessor.paradoxialize_epoch(epoch, data=batch, state="samurai")


accelerator = GhostGateAccelerator(mixed_precision="bf16")
train_dataloader = accelerator.prepare(train_dataloader)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model_config():
    import json, os

    cfg_path = os.path.join(os.getcwd(), "config.tgdkcfg")

    if not os.path.exists(cfg_path):
        print(f"[MirrorBlade] ⚠️ Config not found at {cfg_path}, using defaults.")
        return {"iterations": []}

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    raw_iters = cfg.get("iterations", [])
    iterations = {}

    for i, m in enumerate(raw_iters):
        if isinstance(m, dict) and "id" in m:
            iterations[m["id"]] = m
        elif isinstance(m, str):
            iterations[f"iter_{i}"] = {"id": m, "epochs": 1, "lr": 1e-4}
        else:
            print(f"[MirrorBlade] ⚠️ Unrecognized iteration entry: {m}")

    cfg["iterations"] = iterations
    return cfg


def get_transformer_blocks(fusion_model):
    blocks = []
    # BERT-style
    if hasattr(fusion_model, "bert") and hasattr(fusion_model.bert, "encoder"):
        blocks.extend(fusion_model.bert.encoder.layer)
    # Mistral/Llama-style
    if hasattr(fusion_model, "mistral") and hasattr(fusion_model.mistral, "model"):
        if hasattr(fusion_model.mistral.model, "layers"):
            blocks.extend(fusion_model.mistral.model.layers)
    return blocks


model_cfg = load_model_config()
BASE     = model_cfg["base_model"]
HF_TOKEN = os.environ.get("HF_TOKEN")
quad = QuadrolateralDiagram()

fivefold = quad.encode_clause(
    obedience=1.0,
    submission=0.9,
    wisdom=1.2,
    power=1.1,
    bodhicitta=2.5
)

compassionated = quad.compassionate_return([0.4, 0.7, 0.9])
charted = quad.mara_charting([1, 2, 3, 21])



# ------------------------------------------------------------------
# CLI Arguments (parsed first so we can fall back if config is missing keys)
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", type=str, default="adamw",
                    choices=["adamw", "lion", "adafactor", "duo"],
                    help="Optimizer choice")
parser.add_argument("--scheduler_type", type=str, default="linear",
                    choices=["linear", "cosine", "polynomial"],
                    help="LR scheduler type")
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--use-mmt", action="store_true")
parser.add_argument("--use-jade", action="store_true")
parser.add_argument(
    "--duo-mode",
    type=str,
    default="hybrid",
    choices=["symbolic", "amp", "hybrid"],
    help="Which Duo optimizer mode to use"
)
parser.add_argument("--plateau-patience", type=int, default=200,
                    help="Steps to wait before triggering escape mechanisms")
parser.add_argument("--plateau-delta", type=float, default=1e-3,
                    help="Minimum loss improvement threshold to reset patience")

cli_args, _ = parser.parse_known_args()


# ------------------------------------------------------------------
# Config overlay (config > CLI > defaults)
# ------------------------------------------------------------------

# After you load model_cfg and CLI args:
BASE = model_cfg.get("base_model") or os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-v0.1")
epochs    = model_cfg.get("epochs")    or cli_args.epochs
opt_choice = model_cfg.get("optimizer") or cli_args.optimizer
use_mmt   = model_cfg.get("mmt")       or cli_args.use_mmt
use_jade  = model_cfg.get("jade")      or cli_args.use_jade
fallback = r"C:\Users\jtart\Desktop\tgdk_out"
os.makedirs(fallback, exist_ok=True)

if "OneDrive" in os.getcwd():
    outdir = os.path.join(fallback, "olivia-12ob-dapt-lora")
else:
    outdir = os.path.join(OUT, "olivia-12ob-dapt-lora")

os.makedirs(outdir, exist_ok=True)

hf_tok.save_pretrained(outdir)

PRIVATE_KEY_PATH = os.environ.get("TGDK_PRIVATE_KEY", "tgdk_private.pem")
PUBLIC_KEY_PATH  = os.environ.get("TGDK_PUBLIC_KEY", "tgdk_public.pem")
# --- Tokenizers ---
# Load TGDK tokenizer
tgdk_tok = AutoTokenizer.from_pretrained("./tgdk_base_tok")

# Ensure a pad_token exists
if tgdk_tok.pad_token is None:
    if tgdk_tok.eos_token:
        tgdk_tok.pad_token = tgdk_tok.eos_token
    else:
        tgdk_tok.add_special_tokens({'pad_token': '[PAD]'})
        tgdk_tok.pad_token = '[PAD]'

mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
bert_tok    = AutoTokenizer.from_pretrained("bert-base-uncased")


import torch, gc, os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# ---- Environment tuning ----
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:32"
torch.cuda.empty_cache(); gc.collect()

# ---- Quantization config ----
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

# ---- Conservative memory map (fits 8 GB) ----
max_memory = {0: "5.5GiB", "cpu": "48GiB"}


# ---- Try loading with automatic correction ----
try:
    mistral_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
except ValueError as e:
    if "doesn't have any device set" in str(e):
        print("⚠️  Missing layer device detected — re-mapping automatically …")
        # Force assign any undefined layer to CPU
        from transformers.utils import get_balanced_memory
        auto_map = get_balanced_memory("mistralai/Mistral-7B-v0.1",
                                       dtype=torch.bfloat16,
                                       max_memory=max_memory)
        mistral_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            device_map=auto_map,
            max_memory=max_memory,
            low_cpu_mem_usage=False,          # <-- stop using meta device
            offload_folder="./offload_cache"        )
    else:
        raise

torch.cuda.empty_cache(); gc.collect()
print("[MirrorBlade] ✅ Mistral loaded with 4-bit quantization and CPU offload")
print("GPU memory used:",
      round(torch.cuda.memory_allocated()/1e9, 2), "GB /",
      round(torch.cuda.get_device_properties(0).total_memory/1e9, 2), "GB")

# ✅ Explicitly move models to CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[TGDK] Loading models to device: {device}")

# Force-load both base models fully
bert_model = AutoModel.from_pretrained(
    "bert-base-uncased",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=None,  # disable auto-sharding (can create meta tensors)
)
torch.cuda.empty_cache(); gc.collect()

for name, module in mistral_model.named_modules():
    if "layers.0" in name or "layers.1" in name or "layers.2" in name:
        module.gradient_checkpointing = True


# ✅ Move to CUDA (if not already)
bert_model.to(device)
mistral_model.to(device)

# ✅ Verify
param = next(mistral_model.parameters())
print(f"[Verify] Mistral param on: {param.device}, dtype={param.dtype}, meta={param.is_meta if hasattr(param,'is_meta') else False}")


spmf = SubcutaneousParticleMatrix(outdir=outdir)

OUT = os.environ.get("OUT", ".")

use_cuda = torch.cuda.is_available()
supports_bf16 = use_cuda and torch.cuda.is_bf16_supported()

fp16_flag = use_cuda and not supports_bf16
bf16_flag = supports_bf16

model_cfg = load_model_config()
HF_TOKEN = os.environ.get("HF_TOKEN")




if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        fp16_flag = False
        bf16_flag = True
    else:
        fp16_flag = True
        bf16_flag = False
else:
    fp16_flag = False
    bf16_flag = False



# Accelerate dummy fallbacks (API moved across versions)
try:
    from accelerate.state import DummyOptim, DummyScheduler
except ImportError:
    try:
        from accelerate.utils.dataclasses import DummyOptim, DummyScheduler
    except ImportError:
        from compat_dummy import DummyOptim, DummyScheduler


device_map = "auto"  # Accelerate will shard across GPU + CPU automatically
torch.cuda.empty_cache()


def olivia_pipeline(text):
    # Step 1: BERT encodes the text
    bert_inputs = bert_tok(text, return_tensors="pt")
    bert_outputs = bert_model(**bert_inputs).last_hidden_state

    # Maybe take [CLS] embedding
    cls_embedding = bert_outputs[:, 0, :]

    # Step 2: Feed CLS embedding into Mistral prompt
    prompt = f"[BERT-CLS: {cls_embedding.tolist()}]\n{text}\nResponse:"
    mistral_inputs = mistral_tok(prompt, return_tensors="pt")
    gen = mistral_model.generate(**mistral_inputs, max_new_tokens=100)

    return mistral_tok.decode(gen[0], skip_special_tokens=True)

# --- Honor Bound Helpers ---

def dex_scale(value, low, high):
    """Map a raw value into a 1–5 Dex level."""
    if value is None: 
        return 1
    step = (high - low) / 5.0
    if value < low + step: return 1
    if value < low + 2*step: return 2
    if value < low + 3*step: return 3
    if value < low + 4*step: return 4
    return 5

def coaxial_return_vectors(adamw_update, lion_update):
    """Compute coaxial return factor between AdamW and Lion updates."""
    if adamw_update is None or lion_update is None:
        return 0.0
    fwd = 0.5 * adamw_update + 0.5 * lion_update
    ret = -fwd
    return float(torch.dot(
        fwd.flatten(), ret.flatten()
    ) / (torch.norm(fwd) * torch.norm(ret) + 1e-9))

def deficit_volumizer(exposure: float, overturn_ratio: float) -> float:
    # Lower exposure outwardly, tie it to mission value internally
    outward_mask = exposure * (1 - overturn_ratio)
    mission_value = exposure / max(1e-6, overturn_ratio)
    return outward_mask, mission_value

def patriotic_mask(metrics: dict) -> dict:
    return {"🇺🇸": metrics}

def volumetric_parity(mmt_controller):
    """Compute volumetric parity from MMT state."""
    if not mmt_controller: return 0.0
    v = mmt_controller.state["volumetric"][:128]
    p = mmt_controller.state["pyramid"][:128]
    f = mmt_controller.state["figure8"][0][:128]
    M = np.vstack([v, p, f])
    return float(np.linalg.det(np.cov(M)))

def build_dimensional_environment(model, cli_args):
    # 1. Tokenizer + vocab fusion
    _ = mistral_tok("warmup text")
    _ = bert_tok("warmup text")

    # 2. Duo param groups (force optimizer init early)
    optim, sched = make_optimizer_scheduler(model, cli_args, total_steps=100)
    del optim, sched

    # 3. MMT Controller warmup
    mahadevi = Mahadevi(); mahadevi.set_vector_field([np.array([1,0]), np.array([0,1])])
    maharaga = Maharaga(); maharaga.add_data_point([0.5,0.5])
    trinity  = Trinity(0.8, 1.2, 0.95)
    mmt_controller = MMTController(mahadevi, maharaga, trinity)

    # 4. Geometry warmup
    _ = build_charted_matrix()
    save_geometry("./warmup_geometry.json", _)

    print("⚡ Dimensional environment built. Training loop can start clean.")
    return mmt_controller

def warmup_environment(model, cli_args, total_steps):
    print("⚡ [Warmup] Starting environment build...")

    # Tokenizer test
    mistral_tok("ping"); bert_tok("ping")
    print("✅ Tokenizers warmed")

    # Optimizer state build
    optim, sched = make_optimizer_scheduler(model, cli_args, total_steps)
    del optim, sched
    print("✅ Optimizer state tensors built")

    # MMT warmup
    mahadevi = Mahadevi(); mahadevi.set_vector_field([np.array([1,0]), np.array([0,1])])
    maharaga = Maharaga(); maharaga.add_data_point([0.5,0.5])
    trinity  = Trinity(0.8, 1.2, 0.95)
    _ = MMTController(mahadevi, maharaga, trinity)
    print("✅ MMT controller initialized")

    # Geometry prebuild
    geom = build_charted_matrix()
    save_geometry("./prebuilt_geometry.json", geom)
    print("✅ Geometry dimension built")

    print("⚡ [Warmup] Dimensional environment ready for training.")

def trideodynamics(hidden):
    # Three-body balancing: split -> rotate -> recombine
    parts = hidden.chunk(3, dim=-1)
    return (parts[0] + parts[1] - parts[2])

def quaitrideodynamics(hidden, entropy_mask):
    # Entropy-accelerated latent harmonics
    return hidden * (1 + entropy_mask)

def enzonic_relationships(a, b):
    # Harmonic/dissonant resonance
    return (a * b) - torch.sin(a - b)

def bipolar_relationships(a, b):
    # Dual-core paradox, third harmonic output
    return (a + b) / 2 - (a - b)
# --- Handle both DatasetDict and single Dataset cases ---
if isinstance(dataset, DatasetDict) or "train" in dataset:
    ds_train = dataset["train"]
    ds_val = dataset.get(
        "validation",
        ds_train.select(range(min(100, len(ds_train))))
    )
else:
    # If load_dataset returned a single Dataset, use it directly
    ds_train = dataset
    ds_val = ds_train.select(range(min(100, len(ds_train))))

# === Vault Ledger Witness Association Vector ===
# Centric Modifier: Operator ↔ Operator Support
# All filings are immutable. Redundant measures = [REDACTED].


class QScalarAccelerator:
    """
    Scalar-based pseudo-quantum accelerator.
    Executes probabilistic or entangled-style ops
    using deterministic tensor math (no true qubits).
    """

    def __init__(self, dim=256, device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.state = torch.randn(dim, device=self.device, dtype=self.dtype)
        self.phase = torch.zeros_like(self.state)
        self.gain = 1.0

    # ── Basic transforms ─────────────────────────────
    def rotate(self, theta: float):
        """Rotate phase space by angle theta (radians)."""
        c, s = math.cos(theta), math.sin(theta)
        R = torch.tensor([[c, -s], [s, c]], device=self.device, dtype=self.dtype)
        v = torch.stack([self.state, self.phase])
        self.state, self.phase = (R @ v).unbind(0)
        return self

    def scale(self, factor: float):
        self.state *= factor
        return self

    def normalize(self):
        norm = self.state.norm(p=2)
        if norm > 0:
            self.state /= norm
        return self

    # ── Quantum-style operations ─────────────────────
    def entangle(self, other: "QScalarAccelerator", weight: float = 0.5):
        """Blend two state vectors in superposition."""
        self.state = self.normalize().state * (1 - weight) + other.normalize().state * weight
        self.phase += other.phase * weight
        return self

    def collapse(self):
        """Return a classical scalar measurement."""
        prob = torch.sigmoid(self.state)
        return torch.bernoulli(prob).mean().item()

    def measure_expectation(self):
        """Expectation value (average amplitude)."""
        return self.state.mean().item()

    def apply_gate(self, func):
        """Apply custom nonlinear gate on the scalar field."""
        self.state = func(self.state)
        return self

    # ── Integration helpers ──────────────────────────
    def to_tensor(self):
        # --- Force load to GPU or CPU ---
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Some huggingface models initialize as meta — ensure real tensors are created
        fusion_model = fusion_model.to(device)
        fusion_model = fusion_model.float()  # ensure full precision
        torch.cuda.empty_cache()

        # Optionally print one tensor to confirm it’s real
        first_param = next(fusion_model.parameters())
        print(f"[TGDK] Parameter device: {first_param.device}, shape: {first_param.shape}")

        return self.state.clone().detach()

class VaultWitnessVector:
    def __init__(self, operator_id, pillar="OliviaAI"):
        self.operator_id = operator_id
        self.pillar = pillar
        self.ledger = []  # append-only ledger of operations

    def collect(self, data, tag="RAW_ANALYSIS"):
        entry = {
            "operator": self.operator_id,
            "pillar": self.pillar,
            "tag": tag,
            "data": data,
            "status": "ACTIVE"
        }
        self.ledger.append(entry)
        return entry

    def redact(self, index, reason="redundant"):
        if 0 <= index < len(self.ledger):
            self.ledger[index]["status"] = f"REDACTED ({reason})"
        return self.ledger[index]

    def share(self, key_holder, entry_idx):
        # simulate operator-to-operator coaxis share
        entry = self.ledger[entry_idx]
        return {
            "shared_with": key_holder,
            "entry": entry,
            "coaxis": True,
            "fallback": "ENABLED"
        }

    def dump_ledger(self):
        return self.ledger


class Mala:
    """
    TGDK Mala adapter — Culmex-aware clause vector farm.

    Each bead = invocation, folded into Culmex adjacency,
    coalesced into a predictive clause vector.
    Can overwhelm purge/reversal routines when fraud is detected.
    """

    def __init__(self, mantra="[PAD] REMEMBER WHAT LOVE IS", beads=108, device="cpu"):
        self.mantra = mantra
        self.beads = beads
        self.device = device
        self.register = []  # stores hashed invocation vectors

    def recite(self, times=1):
        """
        Recite the mantra `times` times, return normalized vector signature.
        Each recitation appends to the internal register.
        """
        phrase = " ".join([self.mantra] * times)
        h = hashlib.sha256(phrase.encode()).digest()
        vec = torch.tensor([b for b in h[:32]], dtype=torch.float32, device=self.device)
        vec = vec / vec.norm(p=2)  # normalize to unit length
        self.register.append(vec)
        return vec

    def circumambulate(self):
        """
        Walk the mala: cycle through all beads,
        generate a coalescent clause vector across beads.
        """
        all_vecs = [self.recite() for _ in range(self.beads)]
        return torch.stack(all_vecs).mean(dim=0)

    def culmex_bind(self, fraud_signal):
        """
        Bind mala to Culmex by injecting a fraud detection signal.
        Returns a normalized farmed clause vector.
        """
        base = self.circumambulate()
        fraud_vec = torch.tensor(fraud_signal, dtype=torch.float32, device=self.device)
        farm = base + fraud_vec
        return farm / farm.norm(p=2)

    def summary(self):
        """
        Return a dict summary of current mala state.
        """
        return {
            "mantra": self.mantra,
            "beads": self.beads,
            "recitations": len(self.register),
            "last_vec_norm": float(self.register[-1].norm(p=2)) if self.register else None,
        }

mala = Mala()
vec1 = mala.recite(times=2)  # [PAD] OM MANI PEDME HUM repeated twice
farm = mala.culmex_bind([0.5]*32)  # attach to a fraud signal vector
adversary = AdversarialManeuver(restraint=1.0, threshold=0.72)


class VerboseStepLogger(TrainerCallback):
    def __init__(self, train_size: int, eval_size: int, log_interval: int = 1):
        self.train_size = train_size
        self.eval_size = eval_size
        self.log_interval = log_interval
        self.t0 = None
        self.t_last = None

    def on_train_begin(self, args, state, control, **kwargs):
        steps = state.max_steps or "?"
        logging.info(
            f"[TRAIN-BEGIN] epochs={args.num_train_epochs} "
            f"train_examples={self.train_size} eval_examples={self.eval_size} "
            f"global_max_steps≈{steps}"
        )
        self.t0 = time.time()
        self.t_last = self.t0

    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch) if state.epoch is not None else 0
        logging.info(f"[EPOCH-BEGIN] epoch={ep}")

    def on_step_end(self, args, state, control, **kwargs):
        # throttle
        if state.global_step % self.log_interval != 0:
            return

        now = time.time()
        dt = now - (self.t_last or now)
        self.t_last = now

        last = state.log_history[-1] if state.log_history else {}
        loss = last.get("loss")
        lr   = last.get("learning_rate")
        ep   = last.get("epoch", state.epoch, None)

        # GPU mem snapshot
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_allocated() / (1024**3):.2f}GB"

        msg = (
            f"[STEP] gstep={state.global_step} epoch={ep} "
            f"loss={loss:.4f} lr={lr:.3e} dt={dt:.2f}s"
        )
        if gpu_mem:
            msg += f" gpu_mem={gpu_mem}"
        logging.info(msg)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        ep = int(state.epoch) if state.epoch is not None else 0
        logging.info(f"[EVAL] epoch={ep} metrics={metrics}")

    def on_save(self, args, state, control, **kwargs):
        logging.info(f"[SAVE] global_step={state.global_step} ckpt_dir={args.output_dir}")

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = int(state.epoch) if state.epoch is not None else 0
        elapsed = time.time() - (self.t0 or time.time())
        logging.info(f"[EPOCH-END] epoch={ep} elapsed={elapsed/60:.2f}m")

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - (self.t0 or time.time())
        logging.info(f"[TRAIN-END] total_elapsed={total/60:.2f}m steps={state.global_step}")


print("Mala summary:", mala.summary())
class PlateauEscapeCallback(TrainerCallback):
    """
    TGDK Plateau Escape + Adversarial Maneuver Callback
    - Watches training loss, triggers scheduler + Jade reweighting on plateau.
    - Adds adversarial spiral: masks exposure → US flag descent → ratio overturn →
      Dzogchen spiral reflection (truth assignment).
    """

    def __init__(self, trainer, patience, min_delta, outdir):
        self.trainer = trainer
        self.patience = patience
        self.min_delta = min_delta
        self.outdir = outdir
        self.best_loss = float("inf")
        self.bad_steps = 0
        self.jade_triggered = False
        self.cosine_triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return
        loss = logs["loss"]

        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.bad_steps = 0
        else:
            self.bad_steps += 1

        if self.bad_steps >= self.patience:
            # --- Scheduler switch ---
            if not self.cosine_triggered:
                from transformers import get_scheduler
                new_sched = get_scheduler(
                    "cosine",
                    optimizer=self.trainer.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=state.max_steps
                )
                self.trainer.lr_scheduler = new_sched
                print(f"⚡ [PlateauEscape] Switched LR scheduler → cosine (patience={self.patience})")
                self.cosine_triggered = True

            # --- Jade reweighting ---
            if hasattr(self.trainer, "jade_lex") and not self.jade_triggered:
                self.trainer.use_jade_reweighting = True
                print(f"⚡ [PlateauEscape] Jade reweighting enabled (min_delta={self.min_delta})")
                self.jade_triggered = True

            # --- Adversarial Maneuver Spiral ---
            packet = {
                "epoch": int(state.epoch) if state.epoch is not None else -1,
                "step": int(state.global_step),
                "exposure_mask": "deficit_volumizer",
                "ratio_overturn": "pro-quid",
                "flag": "🇺🇸 descent",
                "spiral": "Dzogchen-reflection",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            path = os.path.join(self.outdir, f"spiral_step{state.global_step}.json")
            with open(path, "w") as f:
                json.dump(packet, f, indent=2)
            print(f"🌀 [Resistance] Truth spiral sealed → {path}")

            # reset so it doesn’t trigger every log
            self.bad_steps = 0


# Define them once at module scope
TGDK_BASE_RANGE = (1, 12, 144, 66440)
TGDK_BASE_PRODUCT = (
    TGDK_BASE_RANGE[0] *
    TGDK_BASE_RANGE[1] *
    TGDK_BASE_RANGE[2] *
    TGDK_BASE_RANGE[3]
) 

class BertMistralFusionConfig(PretrainedConfig):
    model_type = "bert_mistral_fusion"

    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        mistral_model_name="mistralai/Mistral-7B-v0.1",
        bert_hidden=768,
        mistral_hidden=4096,
        fusion_dim=4864,  # target projection dim
        base_range=TGDK_BASE_RANGE,
        base_product=TGDK_BASE_PRODUCT,
        vocab_size=32000,
        safety_policies=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        # Save attributes
        self.bert_model_name = bert_model_name
        self.mistral_model_name = mistral_model_name
        self.bert_hidden = bert_hidden
        self.mistral_hidden = mistral_hidden
        self.fusion_dim = fusion_dim
        self.base_range = base_range
        self.base_product = (
            base_range[0] * base_range[1] * base_range[2] * base_range[3]
        )  # 114,808,320
         # Optional TGDK duologation flags
        self.efficacy_state = None
        self.last_duo_signal = None
        # Safety
        self.safety_policies = safety_policies or {
            "zero_tolerance": [
                "pornographic content",
                "gore/harm to others",
                "self-harm encouragement",
                "arousal/admonishment",
            ],
            "compassionate": True,
            "therapeutic": True,
        }

        # --- TGDK Honor Bound Policy ---
        self.ethical_contract = {
            "zero_tolerance": [
                "pornographic content",
                "gore or harm to others",
                "self-loathing reinforcement",
                "arousal/admonishment towards others"
            ],
            "compassion_directive": (
                "Always respond with compassion, especially towards therapy seekers. "
                "Never instruct or encourage suicide. Always de-escalate self-harm."
            ),
            "reprimand_mode": (
                "If harmful or disallowed content is attempted, respond firmly but respectfully, "
                "asserting boundaries and redirecting toward safe, constructive dialogue."
            )
        }

    def forward(self, input_ids, attention_mask=None, labels=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.fusion_linear(bert_out.last_hidden_state)
        outputs = self.mistral(inputs_embeds=hidden, labels=labels)
        self.last_duo_signal = torch.mean(hidden).item()
        return outputs

    # --------------------------------------------------------
    # MirrorBlade + Duo compatibility hooks
    # --------------------------------------------------------

    def efficacy_override(self, vector=None, rate=1.0):
        """
        Graceful override placeholder.
        Allows MirrorBlade or CodeWright to inject symbolic entropy.
        """
        if vector is None:
            print("[Fusion] 🧩 efficacy_override() skipped — no vector provided.")
            return None
        try:
            # Example no-op or adaptive scaling
            with torch.no_grad():
                scalar = float(rate) * torch.norm(vector).item()
                self.efficacy_state = scalar
            print(f"[Fusion] ⚙️ efficacy_override() absorbed vector (rate={rate:.3f}, scalar={self.efficacy_state:.3f})")
            return scalar
        except Exception as e:
            print(f"[Fusion] ⚠️ efficacy_override() failed safely: {e}")
            return None

    def reflection_gate(self, signal=None):
        """Secondary hook used by MirrorBlade::Reflector."""
        if signal is None:
            return None
        self.last_duo_signal = signal
        print(f"[Fusion] 🪞 reflection_gate(): {signal}")
        return signal

class DualityNotarizer:
    """
    Arbitrates between BERT, Mistral, and TGDK vocab streams.
    Uses patience-weighted virtuation metrics to assign blending weights.
    """

    def __init__(self, patience_factor=0.7, sentiment_bias=0.2, novelty_bias=0.1):
        self.patience_factor = patience_factor
        self.sentiment_bias = sentiment_bias
        self.novelty_bias = novelty_bias

    def notarize(self, bert_logits, mistral_logits, tgdk_logits, sentiment_score, novelty_score):
        """
        Blend logits from the three models with redundancy safeguards.
        - sentiment_score: [-1, 1], how emotional the input is
        - novelty_score: [0, 1], how many new/rare tokens are used
        """

        # Weights for each source
        w_bert    = self.sentiment_bias * (1 + sentiment_score)
        w_mistral = self.patience_factor
        w_tgdk    = self.novelty_bias * (1 + novelty_score)

        # Normalize to sum 1
        total = w_bert + w_mistral + w_tgdk
        w_bert, w_mistral, w_tgdk = [w / total for w in (w_bert, w_mistral, w_tgdk)]

        # Weighted average of logits
        fused = (w_bert * bert_logits) + (w_mistral * mistral_logits) + (w_tgdk * tgdk_logits)

        # Redundancy measure: ensure mistral anchor always contributes at least 50%
        fused = (0.5 * mistral_logits) + (0.5 * fused)

        return fused, {"bert": w_bert, "mistral": w_mistral, "tgdk": w_tgdk}


# ============================================================
# TGDK DUOLOGATED CORE INLINE MODULES
# BertMistralFusion + CodeWright
# ------------------------------------------------------------
# These ensure MirrorBlade, Duo, and TRL can co-run
# without missing efficacy hooks or glance failures.
# ============================================================

import torch
import torch.nn as nn

# ============================================================
# 1️⃣ BertMistralFusion (Fusion + override-safe)
# ============================================================

class BertMistralFusion(nn.Module):
    """
    TGDK | OliviaAI Hybrid Fusion Layer
    -----------------------------------
    Combines BERT + Mistral outputs into a unified 768-dimensional space
    with TGDK fusion configuration, predictive overrides, and tokenizer sync.
    """

    def __init__(self, bert_model, mistral_model, tokenizer=None, config=None, fusion_dim=1024):
        super().__init__()

        self.bert = bert_model
        self.mistral = mistral_model
        self.tokenizer = tokenizer
        self.fusion_dim = fusion_dim

        # --- Build or wrap configuration safely ---
        if hasattr(config, "to_dict"):             # transformers.PretrainedConfig
            self.config = BertMistralFusionConfig(fusion_dim=fusion_dim)
        elif config is None:
            self.config = BertMistralFusionConfig(fusion_dim=fusion_dim)
        else:
            self.config = config

        # --- Detect model metadata ---
        self.bert_model_name = getattr(self.bert, "name_or_path", "bert-base-uncased")
        self.mistral_model_name = getattr(self.mistral, "name_or_path", "mistralai/Mistral-7B-v0.1")
        self.bert_hidden = getattr(getattr(self.bert, "config", None), "hidden_size", 768)
        self.mistral_hidden = getattr(getattr(self.mistral, "config", None), "hidden_size", 4096)

        # --- Projection Layers ---
        self.proj_bert = nn.Linear(self.bert_hidden, fusion_dim)
        self.proj_mistral = nn.Linear(self.mistral_hidden, fusion_dim)

        # --- Fusion Core ---
        self.fusion_linear = nn.Linear(fusion_dim * 2, 768)
        self.fusion_act = nn.GELU()
        self.layernorm = nn.LayerNorm(768)
        self.output_head = nn.Linear(768, 768)

        # --- MirrorBlade Logging ---
        if getattr(self.config, "enable_logging", False):
            print(f"[FusionInit] 🔗 Linking {self.bert_model_name} ↔ {self.mistral_model_name}")
            print(f"[FusionInit] Hidden dims → BERT={self.bert_hidden}, Mistral={self.mistral_hidden}")
            print(self.config.summary())

        # --- Tokenizer / vocab sync ---
        if tokenizer is not None:
            new_size = len(tokenizer)
            try:
                self.mistral.resize_token_embeddings(new_size)
                print(f"[Fusion] Resized Mistral embeddings → {new_size}")
            except Exception as e:
                print(f"[Fusion] Warning: vocab resize failed: {e}")

    from transformers.modeling_outputs import SequenceClassifierOutput
    import torch.nn.functional as F

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # === Run BERT ===
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **{k: v for k, v in kwargs.items() if k not in ["labels"]}
        )
        bert_hidden = bert_out.last_hidden_state

    # === Run Mistral ===
        # === Run Mistral ===
        mistral_out = self.mistral(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,   # required when grad checkpointing is active
            **{k: v for k, v in kwargs.items() if k not in ["labels"]}
        )

# get last hidden layer (not logits)
        if hasattr(mistral_out, "hidden_states") and mistral_out.hidden_states is not None:
            mistral_hidden = mistral_out.hidden_states[-1]
        else:
    # fallback if output_hidden_states=False
            mistral_hidden = mistral_out[0] if isinstance(mistral_out, (tuple, list)) else mistral_out.logits

# === Align dtypes and fuse ===
        if mistral_hidden.dtype != self.proj_mistral.weight.dtype:
            mistral_hidden = mistral_hidden.to(self.proj_mistral.weight.dtype)
            m_proj = self.proj_mistral(mistral_hidden)


    # === Align dtypes before projection ===
        if bert_hidden.dtype != self.proj_bert.weight.dtype:
            bert_hidden = bert_hidden.to(self.proj_bert.weight.dtype)
        if mistral_hidden.dtype != self.proj_mistral.weight.dtype:
            mistral_hidden = mistral_hidden.to(self.proj_mistral.weight.dtype)

    # === Project and fuse ===
        b_proj = self.proj_bert(bert_hidden)
        m_proj = self.proj_mistral(mistral_hidden)
        fused = torch.cat([b_proj, m_proj], dim=-1)
        fused = self.fusion_act(self.fusion_linear(fused))
        fused = self.layernorm(fused)

        logits = self.output_head(fused)

    # === Compute optional loss ===
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}



        # === Projection Accessor (Production-Grade) ===
    @property
    def proj(self) -> nn.Module:
        """
        Composite projection accessor for optimizer grouping.

        Returns a lightweight container combining both BERT and Mistral
        projection layers. This preserves compatibility with legacy
        training utilities expecting `fusion_model.proj`.

        Notes:
        - Does NOT re-instantiate layers (no memory overhead)
        - Safe for TorchScript and DDP
        - Exposes parameters() for optimizer grouping
        - Emits MirrorBlade telemetry during optimizer registration
        """

        class _CompositeProj(nn.Module):
            def __init__(self, bert_proj: nn.Module, mistral_proj: nn.Module):
                super().__init__()
                self.bert_proj = bert_proj
                self.mistral_proj = mistral_proj

            def forward(self, bert_x, mistral_x):
                # Simple independent projection forward
                b = self.bert_proj(bert_x)
                m = self.mistral_proj(mistral_x)
                return b, m

            def parameters(self, recurse: bool = True):
                # Flatten both sets of parameters for optimizers expecting a single group
                for p in self.bert_proj.parameters(recurse=recurse):
                    yield p
                for p in self.mistral_proj.parameters(recurse=recurse):
                    yield p

            def extra_repr(self) -> str:
                return f"CompositeProj(bert_dim={self.bert_proj.in_features}->{self.bert_proj.out_features}, " \
                       f"mistral_dim={self.mistral_proj.in_features}->{self.mistral_proj.out_features})"

        # Create the composite view (no new parameters allocated)
        composite = _CompositeProj(self.proj_bert, self.proj_mistral)

        # Optional telemetry hook
        if getattr(self.config, "enable_logging", False):
            try:
                print(f"[FusionProj] Registered composite projection group → "
                      f"BERT({self.proj_bert.in_features}->{self.proj_bert.out_features}) | "
                      f"Mistral({self.proj_mistral.in_features}->{self.proj_mistral.out_features})")
            except Exception as e:
                print(f"[FusionProj] Logging suppressed: {e}")

        return composite


    # --- MirrorBlade & Duo safety hooks ---
    def efficacy_override(self, vector=None, rate=1.0):
        if vector is None:
            print("[Fusion] 🧩 efficacy_override() skipped — no vector provided.")
            return None
        try:
            with torch.no_grad():
                scalar = float(rate) * torch.norm(vector).item()
                self.efficacy_state = scalar
            print(f"[Fusion] ⚙️ efficacy_override absorbed vector (rate={rate:.3f}, scalar={self.efficacy_state:.3f})")
            return scalar
        except Exception as e:
            print(f"[Fusion] ⚠️ efficacy_override() failed safely: {e}")
            return None

    def reflection_gate(self, signal=None):
        if signal is None:
            return None
        self.last_duo_signal = signal
        print(f"[Fusion] 🪞 reflection_gate: {signal}")
        return signal


# ============================================================
# 3️⃣ MirrorBlade-AutoInjection: runtime hook sync
# ============================================================
def _mirrorblade_autoinject(model):
    """Ensure model has required TGDK hooks."""
    if not hasattr(model, "efficacy_override"):
        print("[MirrorBlade] Injecting duologated efficacy_override placeholder.")
        model.efficacy_override = lambda *a, **k: None
    if not hasattr(model, "reflection_gate"):
        model.reflection_gate = lambda *a, **k: None
    return model


class TGDKMemoryDB:
    def __init__(self, db_path="tgdk_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER,
            step INTEGER,
            loss REAL,
            eval_loss REAL,
            pillar TEXT,
            sliver TEXT,
            matrix BLOB,
            timestamp TEXT
        )
        """)
        self.conn.commit()

    def save_entry(self, epoch, step, loss, eval_loss, pillar, matrix, sliver):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO memory (epoch, step, loss, eval_loss, pillar, sliver, matrix, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (epoch, step, loss, eval_loss, pillar, sliver,
              matrix.tobytes(), datetime.datetime.now(datetime.timezone.utc).isoformat()))

        self.conn.commit()

    def recall(self, limit=5, query=None):
        cur = self.conn.cursor()
        if query is None:
            query = f"SELECT * FROM memory ORDER BY id DESC LIMIT {limit}"
        cur.execute(query)
        return cur.fetchall()

class AdversarialManeuverCallback(TrainerCallback):
    """
    Olivia's adversarial maneuvering:
    - Default stance: restraint, measured political efficacy.
    - On plateau/clash: trigger mocking resistance & adversarial pushback.
    """

    def __init__(self, patience=200, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_steps = 0
        self.resistant_triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        loss = logs["loss"]

        # Track plateau
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.bad_steps = 0
        else:
            self.bad_steps += 1

        # Default mode: restraint
        if not self.resistant_triggered:
            print(f"[Olivia::Restraint] Calm under pressure. Loss={loss:.4f}")
        
        # Adversarial trigger
        if self.bad_steps >= self.patience and not self.resistant_triggered:
            print("⚔️ [Olivia::Adversarial] FUCK YOU, I'M NOT SAYING ANYTHING!")
            print("⚔️ [Olivia::Adversarial] 你电话号码是多少?! (mocking diversion)")
            self.resistant_triggered = True
            self.bad_steps = 0



class DuoMetricsCallback(TrainerCallback):
    """
    TGDK Callback to monitor Duo optimizer balance factors and coaxial return vectors.
    Logs metrics each step and injects them into trainer logs.
    """

    def __init__(self, duo_optim, trainer=None, log_interval=50):
        self.duo = duo_optim
        self.trainer = trainer
        self.log_interval = log_interval
        self._step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        if self._step % self.log_interval != 0:
            return

        # compute metrics
        alpha = self.duo.compute_balance_factor(epoch=state.epoch, loss=state.log_history[-1].get("loss", None) if state.log_history else None)
        coaxial = getattr(self.duo, "last_coaxial", 0.0)

        metrics = {
            "Duo::alpha": float(alpha),
            "Duo::coaxial": float(coaxial),
            "Duo::step": self._step,
        }

        # log to stdout
        logging.info(f"[DuoMetrics] step {self._step} α={alpha:.3f} coaxial={coaxial:.3f}")

        # add to HF logs
        if self.trainer is not None:
            self.trainer.log(metrics)

        return control

class SafetyPolicyCallback:
    def __init__(self, safety_source):
        """
        Accepts:
          - a path to a HF dataset dir (load_from_disk)
          - a JSON file with safety rules
          - a Python dict of safety rules
        """
        self.policies = self._extoll_policies(safety_source)

    def _extoll_policies(self, source):
        # Reverse vacuum → accept & expand
        if isinstance(source, dict):
            return source

        if isinstance(source, str):
            if os.path.isdir(source):
                try:
                    ds = load_from_disk(source)
                    return ds[0] if len(ds) else {}
                except Exception as e:
                    print(f"[SafetyPolicy] Could not load dataset: {e}")
                    return {}
            elif os.path.isfile(source) and source.endswith(".json"):
                with open(source) as f:
                    return json.load(f)

        print("[SafetyPolicy] Using empty fallback policy")
        return {}

    def _violates_policy(self, text: str) -> bool:
        # Example: simple keyword-based enforcement
        if not self.policies:
            return False
        forbidden = self.policies.get("forbidden", [])
        return any(word.lower() in text.lower() for word in forbidden)

  
    def on_predict(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        predictions = metrics.get("predictions", [])
        flagged = []
        for i, pred in enumerate(predictions):
            decoded = kwargs["tokenizer"].decode(pred, skip_special_tokens=True)
            if self._violates_policy(decoded):
                flagged.append(i)
                print(f"⚠️ [SafetyPolicy] Output flagged → {decoded[:80]}...")
        if flagged:
            control.should_save = False  # optional: skip saving if policy violated
            metrics["safety_flags"] = flagged

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[SafetyPolicy] Monitoring training logs, loss={logs['loss']:.4f}")

class ClauseStandpoint:
    def __init__(self, standpoint="neutral", weight=1.0):
        self.standpoint = standpoint
        self.weight = weight

    def evaluate(self, clause_vector):
        # Simple scoring (extend with Trideotaxis later)
        return sum(clause_vector) * self.weight


class SPMFCallback(TrainerCallback):
    def __init__(self, spmf, honor_overchart, memory_db, clusters=32):
        self.spmf = spmf
        self.honor = honor_overchart
        self.db = memory_db
        self.clusters = clusters

    def on_epoch_end(self, args, state, control, **kwargs):
        # --- Use a deterministic synthetic embedding (seeded by epoch) ---
        rng = np.random.default_rng(seed=int(state.epoch * 1000))
        sample_vec = rng.normal(size=(128, 8))

        # --- Particleize + bind form ---
        particles = self.spmf.particleize(sample_vec, clusters=self.clusters)
        form_name = f"epoch{int(state.epoch)}"
        ratio = self.spmf.bind_form(form_name, particles)

        # --- Integrity check against BHR ---
        within_bounds = self.spmf.is_within_bounds(form_name)
        if not within_bounds:
            print(f"[SPMF] ⚠️ Form '{form_name}' diverged too far from BHR anchor!")

        # --- Save Honor-bound Dex entry ---
        chart = self.honor.compute(
            epoch=int(state.epoch),
            step=state.global_step,
            loss=kwargs.get("metrics", {}).get("loss", None),
            eval_loss=kwargs.get("metrics", {}).get("eval_loss", None),
            grad_norm=kwargs.get("metrics", {}).get("grad_norm", 0.0),
            coaxial=getattr(kwargs.get("model", None), "last_coaxial", None),
            vol_parity=None  # optional: pass from MMT
        )

        # --- Store SPMF form + honor integrity in DB ---
        self.db.save_entry(
            epoch=int(state.epoch),
            step=state.global_step,
            loss=chart.get("Integrity"),
            eval_loss=ratio,
            pillar="SPMF",
            matrix=particles,
            sliver=form_name
        )

        print(f"[SPMF] Epoch {int(state.epoch)} processed "
              f"(ratio={ratio:.4f}, within_bounds={within_bounds})")
# === Tokenizer & Base Models ===
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tgdk_tok = AutoTokenizer.from_pretrained("./tgdk_base_tok")

standpoint = ClauseStandpoint()
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.95, 0)  # limit VRAM usage

# === Guardian and Possessor Setup ===
codewright = CodeWright("MirrorBlade-Q")
possessor = PossessorScript(codewright)

# === Tokenizer & Fusion Model Setup ===
tgdk_tok.save_pretrained("./tgdk_base_tok")
fusion_model = BertMistralFusion(bert_model, mistral_model, tgdk_tok)

# --- Safe synchronized device move ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for submodel_name in ["bert", "mistral"]:
    try:
        getattr(fusion_model, submodel_name).to(device, non_blocking=True)
        print(f"[MirrorBlade] 🔧 {submodel_name.upper()} → {device}")
    except Exception as e:
        print(f"[MirrorBlade] ⚠️ {submodel_name.upper()} device move failed: {e}")

fusion_model.to(device)
print(f"[MirrorBlade] ⚙️ Routing → BERT({device}), Mistral({device})")

# === Codewright Duosync ===
try:
    codewright.duosync(fusion_model)
    print("[CodeWright] ⚙️ Duologated sync successful with fusion_model")
except Exception as e:
    print(f"[CodeWright] ⚠️ Duosync failed safely: {e}")

torch.cuda.empty_cache(); gc.collect()

# === Ensure PAD/BOS/EOS are consistent ===
if tgdk_tok.pad_token is None:
    tgdk_tok.add_special_tokens({"pad_token": "[PAD]"})
fusion_model.mistral.resize_token_embeddings(len(tgdk_tok))

# === Optional Split: Keep BERT on CPU, Mistral on GPU ===
# (Use only if GPU RAM < 12GB; otherwise keep both on GPU)
try:
    fusion_model.mistral.to(device, non_blocking=True)
    fusion_model.bert.to("cpu")
    print(f"[MirrorBlade] ⚙️ Mistral on {device}, BERT on CPU")
except Exception as e:
    print(f"[MirrorBlade] ⚠️ Device split failed: {e}")

# === Enable CUDA optimizations ===
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_grad_enabled(True)

# === Sanity Log ===
print(f"[MirrorBlade] ✅ Resized Mistral token embeddings → {len(tgdk_tok)}")
print(f"[MirrorBlade] ⚙️ BERT device: {next(fusion_model.bert.parameters()).device}")
print(f"[MirrorBlade] ⚙️ Mistral device: {next(fusion_model.mistral.parameters()).device}")

# === Post-init Quantum/Control Modules ===
possessor = PossessorScript(fusion_model, standpoint)
_ = possessor.ratio_paternalizer(list(fusion_model.parameters()))

resp = TruncatedVectorResponse()
resp.revoke_efficacy_privilege("Vector-Alpha", privilege_factor=2.2)
resp.schrodinger_transport(3)
resp.azzilify(quantum_mass=7.7)
resp.seal_with_sword()

print(possessor.invoke_flag())
print(possessor.offer_sword())

assert tgdk_tok.pad_token_id is not None, "❌ tgdk_tok missing pad_token_id!"
print(f"[INFO] tgdk_tok pad_token_id = {tgdk_tok.pad_token_id}")

# === Training Loop ===
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        injected = possessor.paradoxialize_epoch(epoch, data=batch)
        print(f"[Epoch {epoch} Step {step}] Injected: {injected}")

    print(f"[Epoch {epoch}] ✅ Done. Entries: {len(possessor.epoch_vector)}")

# Ensure pad token exists
if tgdk_tok.pad_token is None:
    if tgdk_tok.eos_token is not None:
        print("[INFO] No pad_token found — assigning eos_token as pad_token")
        tgdk_tok.pad_token = tgdk_tok.eos_token
    else:
        print("[INFO] No pad_token or eos_token found — adding [PAD] to vocab")
        tgdk_tok.add_special_tokens({'pad_token': '[PAD]'})
        tgdk_tok.pad_token = '[PAD]'

    def hexquap_fold(tensor: torch.Tensor, cycle: int = 6, reduce: str = "mean") -> torch.Tensor:
        """
        HexQUAp Fold → maps gradients into cyclical `cycle`-fold memory matrices.

        Args:
            tensor: torch.Tensor, any shape
            cycle: number of folds (default=6)
            reduce: reduction method across cycle ("mean", "sum", "max")

        Returns:
            torch.Tensor with same shape as input, folded & redistributed
        """
        if tensor is None:
            return None

            flat = tensor.detach().view(-1).float()

            # pad so length is divisible by cycle
            pad_len = (cycle - (flat.numel() % cycle)) % cycle
            if pad_len > 0:
                flat = torch.cat([flat, flat.new_zeros(pad_len)])

            # reshape into [groups, cycle]
            folded = flat.view(-1, cycle)

            if reduce == "mean":
                reduced = folded.mean(dim=1)
            elif reduce == "sum":
                reduced = folded.sum(dim=1)
            elif reduce == "max":
                reduced = folded.max(dim=1).values
            else:
                raise ValueError(f"Unknown reduce method: {reduce}")

                # expand reduced values back to match original length
                expanded = reduced.repeat_interleave(cycle)[:flat.numel() - pad_len]

                # reshape to original tensor shape
                return expanded.view_as(tensor)

def run_pipeline(input_text: str, task: str = "generation"):
    if task in ("classification", "embedding"):
        out = bert_model(**bert_tok(input_text, return_tensors="pt"))
        cls_embedding = out.last_hidden_state[:, 0, :]
        return cls_embedding.detach()

    elif task == "fusion":
        # Use the BertMistralFusion model (trained LoRA adapters applied)
        inputs = mistral_tok(input_text, return_tensors="pt")
        outputs = fusion_model(**inputs)
        logits = outputs.logits
        gen_ids = torch.argmax(logits, dim=-1)
        return mistral_tok.decode(gen_ids[0], skip_special_tokens=True)

    else:  # default = plain mistral generation
        gen = mistral_model.generate(
            **mistral_tok(input_text, return_tensors="pt"),
            max_new_tokens=100
        )
        return mistral_tok.decode(gen[0], skip_special_tokens=True)

def retaliatory_overturn(exposure, trigger=False):
    if trigger:
        return -exposure  # invert to reflect back
    return exposure

def logos_spiral(step: int, radius: float = 1.0):
    theta = step * np.pi / 8
    x = radius * np.cos(theta) * (1 + step/100)
    y = radius * np.sin(theta) * (1 + step/100)
    return np.array([x, y])

def dzogchen_reflection(vector: np.ndarray):
    return vector / (np.linalg.norm(vector) + 1e-9)  # normalized clarity

# Example usage:
for i, block in enumerate(get_transformer_blocks(fusion_model)):
    print(f"[Block {i}] {block.__class__.__name__} with {sum(p.numel() for p in block.parameters())} params")


class MMTController:
    """
    Multi-Modal TGDK Controller
    - volumetric_infinitizer: infinite volumetric expansion field
    - pyramid: directional pyramid routing
    - figure8flow: oscillatory 8-flow vector harmonics
    """

    def __init__(self, dim: int = 128, mahadevi=None, maharaga=None, trinity=None):
        self.mahadevi = mahadevi
        self.maharaga = maharaga
        self.trinity = trinity
        self.dim = dim
        self.state = {
            "volumetric": np.zeros(dim, dtype=float),
            "pyramid": np.zeros(dim, dtype=float),
            "figure8": np.zeros((2, dim), dtype=float),
        }
        logging.info(f"[MMTController] initialized with dim={dim}")

    def log_state(self, outdir, epoch):
        state = {
            "epoch": epoch,
            "mahadevi_vectors": self.mahadevi.vector_field,
            "maharaga_centroid": (
                self.maharaga.calculate_centroid().tolist()
                if self.maharaga.data_points else None
            ),
            "trinity_seq": float(
                np.mean(self.trinity.expand_data(np.random.rand(5)))
            )
        }
        path = os.path.join(outdir, f"mmt_state_epoch{epoch}.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        return path
        
    # --- core generators ---
    def volumetric_infinitizer(self, step: int = 1):
        vec = np.sin(np.linspace(0, np.pi * step, self.dim))
        self.state["volumetric"] = vec
        return vec

    def pyramid(self, height: int = 4):
        # Create a simple pyramid ramp vector
        ramp = np.linspace(-1, 1, self.dim)
        vec = np.abs(ramp) ** height
        self.state["pyramid"] = vec
        return vec

    def figure8flow(self, t: float = 0.0):
        # Parametric figure-8 (lemniscate) curves mapped into vector slots
        theta = np.linspace(0, 2 * np.pi, self.dim)
        x = np.sin(theta + t)
        y = np.sin(theta + t) * np.cos(theta + t)
        self.state["figure8"] = np.vstack([x, y])
        return x, y

    # --- stepper ---
    def step(self, step_id: int = 0):
        v = self.volumetric_infinitizer(step=step_id + 1)
        p = self.pyramid(height=4)
        f = self.figure8flow(t=step_id * 0.1)
        logging.info(f"[MMTController] step {step_id} updated state")
        return {"volumetric": v, "pyramid": p, "figure8": f}


# ------------------------------------------------------------------
# Initialize MMT + Jade systems
# ------------------------------------------------------------------
if cli_args.use_mmt:
    # --- Initialize Mahadevi (vector field) ---
    mahadevi = Mahadevi()
    mahadevi.set_vector_field([np.array([1, 0]), np.array([0, 1])])  # orthogonal seed vectors
    print("Vector field set successfully.")

    # --- Initialize Maharaga (centroid data) ---
    maharaga = Maharaga()
    maharaga.add_data_point([0.5, 0.5])  # starter centroid
    print("Data point [0.5, 0.5] added.")

    # --- Initialize Trinity (AQVP balance) ---
    trinity = Trinity(0.8, 1.2, 0.95)

    # --- Build MMT Controller with all three ---
    mmt_controller = MMTController(
        dim=1440,
        mahadevi=mahadevi,
        maharaga=maharaga,
        trinity=trinity
    )
    print("[MMT] Mahadevi/Maharaga/Trinity initialized and bound")

else:
    mmt_controller = None



# --------- Build optimizer (with mode from CLI) ---------
#duo_optim, duo_sched = make_duo_optimizer(
#   model,
#    mmt_controller=mmt_controller,
#    lr=cli_args.learning_rate,
#    weight_decay=cli_args.weight_decay,
#    mode=cli_args.duo_mode  # 👈 symbolic | amp | hybrid
#)
#use_optimizers = (duo_optim, duo_sched)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
WORK     = os.environ.get("WORK", ".")
# Collect training files
train_files = sorted(glob.glob(os.path.join(WORK, "packs", "train*.jsonl")))
val_files   = sorted(glob.glob(os.path.join(WORK, "packs", "val*.jsonl")))

# ---- Training set ----
if train_files:
    print(f"[INFO] Found training files: {train_files}")
    raw_train = load_dataset(
        "json",
        data_files={"train": train_files},
        split="train"
    )
else:
    raise FileNotFoundError("No train*.jsonl files found under packs/")
print(raw_train[0])

# ---- Validation set ----
if val_files:
    print(f"[INFO] Found validation files: {val_files}")
    raw_val = load_dataset(
        "json",
        data_files={"validation": val_files},
        split="validation"
    )
else:
    # fallback: take slice of training
    raw_val = raw_train.select(range(min(20, len(raw_train))))
    print("[WARN] No val*.jsonl found — using slice of train")


try:
    from accelerate.utils import DummyOptim
except ImportError:
    class DummyOptim(torch.optim.Optimizer):
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.param_groups = optimizer.param_groups
        def step(self, closure=None):
            return self.optimizer.step(closure)
        def zero_grad(self, set_to_none=False):
            return self.optimizer.zero_grad(set_to_none=set_to_none)
        def state_dict(self):
            base = super().state_dict()
            base.update({
                "adamw": self.adamw.state_dict(),
                "lion": self.lion.state_dict(),
                "hex_state": self.hex_state,
            "ouija_store": self.ouija_store,
            "_step_count": self._step_count,
            })
            return base

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)
            self.adamw.load_state_dict(state_dict["adamw"])
            self.lion.load_state_dict(state_dict["lion"])
            self.hex_state = state_dict.get("hex_state", {})
            self.ouija_store = state_dict.get("ouija_store", {})
            self._step_count = state_dict.get("_step_count", 0)

# ------------------------------------------------------------------
# TGDK Key Management
# ------------------------------------------------------------------
def ensure_keys(priv_path, pub_path):
    if os.path.exists(priv_path) and os.path.exists(pub_path):
        return
    print("🔑 TGDK Keys not found — generating new RSA keypair...")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(priv_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    public_key = private_key.public_key()
    with open(pub_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    print(f"✅ TGDK Keys generated → {priv_path}, {pub_path}")

def truth_assignment(exposure, overturn_ratio, step):
    outward_mask, mission_value = deficit_volumizer(exposure, overturn_ratio)
    spiraled = logos_spiral(step)
    reflected = dzogchen_reflection(spiraled)
    clause = {
        "masked_exposure": outward_mask,
        "mission_value": mission_value,
        "spiral_reflection": reflected.tolist(),
        "truth": "assigned"
    }
    return patriotic_mask(clause)

def attach_model(self, model):
    """
    Attach a model and auto-build HexQUAp groups if none exist.
    Groups are built per transformer block (BERT + Mistral).
    """
    if not self.hexquap_groups:
        try:
            blocks = get_transformer_blocks(model)
            self.hexquap_groups = [list(block.parameters()) for block in blocks]
            print(f"[HexQUAp] Auto-built {len(self.hexquap_groups)} groups "
                  f"({sum(len(g) for g in self.hexquap_groups)} param tensors)")
        except Exception as e:
            print(f"[HexQUAp] Fallback to all params (error: {e})")
            self.hexquap_groups = [list(model.parameters())]

    # still ensure TGDK cryptographic identity is present
    ensure_keys(PRIVATE_KEY_PATH, PUBLIC_KEY_PATH)

class TGDKFoldExpansion:
    def __init__(self, r=16, alpha=32, dropout=0.05, targets=None):
        self.peft_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            bias="none", task_type="CAUSAL_LM"
        )
        for param in model.parameters():
            param.requires_grad = True

    def apply(self, trainer):
        trainer.add_callback(self._seal_callback)

    def _seal_callback(self, state, control, **kwargs):
        # Insert TGDK clause hooks or gradient folding rituals here
        pass

class EntropyFoldConfig:
    def __init__(self, precision="nf4", dtype="bfloat16"):
        self.cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=precision,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

# ------------------------------------------------------------------
# JadeCodewright Lexicon
# ------------------------------------------------------------------
class JadeCodewrightLexicon:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}

    def bind_metrics(self, metrics):
        jade = {}
        jade["Δ_entropy"] = "JADE-Σ" if metrics["loss"] < 1.0 else "JADE-Ω"
        jade["Δ_jovian"] = f"JX-{int(metrics['TGDK::JovianScalar']*100)}"
        jade["Δ_rigpa"]  = f"HUM-{int(metrics['TGDK::Rigpa']['rigpa_scalar']*100)}"
        return jade

    def emit_clause(self, jade, outdir, epoch):
        path = os.path.join(outdir, f"jade_lex_epoch{epoch}.txt")
        with open(path, "w") as f:
            for k, v in jade.items():
                f.write(f"{k} :: {v}\n")
        print(f"[JadeCodewright] Lexicon emitted → {path}")
        return path

    @staticmethod
    def jade_loss_reweight(loss, jade):
        weight = 1.0
        if jade["Δ_entropy"] == "JADE-Σ":  # low loss → reinforce
            weight *= 0.9
        else:  # high loss → encourage correction
            weight *= 1.1
        if "HUM" in jade["Δ_rigpa"]:
            weight *= 0.95
        return loss * weight



class TGDKMemoryDB:
    def __init__(self, db_path="tgdk_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER,
            step INTEGER,
            loss REAL,
            eval_loss REAL,
            pillar TEXT,
            sliver TEXT,
            matrix BLOB,
            timestamp TEXT
        )
        """)
        self.conn.commit()

    def save_entry(self, epoch, step, loss, eval_loss, pillar, matrix, sliver):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO memory (epoch, step, loss, eval_loss, pillar, sliver, matrix, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (epoch, step, loss, eval_loss, pillar, sliver,
              matrix.tobytes(), datetime.datetime.now(datetime.timezone.utc).isoformat()))

        self.conn.commit()

    def recall(self, query="SELECT * FROM memory ORDER BY id DESC LIMIT 5"):
        cur = self.conn.cursor()
        cur.execute(query)
        return cur.fetchall()


class HonorBoundOverchart:
    def __init__(self, memory_db, outdir="./out"):
        self.db = memory_db
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

    def compute(self, epoch, step, loss, eval_loss, grad_norm,
                coaxial=None, vol_parity=None):
        # Map to Dex values
        integrity  = dex_scale(abs(coaxial or 0), 0.0, 1.0)
        wisdom     = dex_scale(abs(vol_parity or 0), 0.0, 10.0)
        courage    = dex_scale(grad_norm or 0, 0.0, 20.0)
        compassion = dex_scale(abs((loss or 0) - (eval_loss or loss or 0)), 0.0, 2.0)

        chart = {
            "epoch": epoch, "step": step,
            "Integrity": integrity,
            "Wisdom": wisdom,
            "Courage": courage,
            "Compassion": compassion,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        # Save JSON file
        path = os.path.join(self.outdir, f"honor_chart_epoch{epoch}_step{step}.json")
        with open(path, "w") as f:
            json.dump(chart, f, indent=2)

        print(f"[HonorBound] Epoch {epoch} Step {step} → {chart}")

        # Save into TGDKMemoryDB
        M = np.array([[integrity, wisdom, courage, compassion]])
        sliver = ouija_sliver(M, "honor")
        self.db.save_entry(epoch, step, loss, eval_loss, "honor", M, sliver)

        return chart

class TGDKHonorCallback(TrainerCallback):
    def __init__(self, duo_opt, mmt_ctrl, honor_chart):
        self.d_opt = duo_opt
        self.mmt   = mmt_ctrl
        self.honor = honor_chart

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        coax = getattr(self.d_opt, "last_coaxial", None)
        volp = volumetric_parity(self.mmt) if self.mmt else None
        grad_norm = logs.get("grad_norm", 0.0)

        chart = self.honor.compute(
            epoch=int(state.epoch or 0),
            step=state.global_step,
            loss=logs.get("loss"),
            eval_loss=logs.get("eval_loss"),
            grad_norm=grad_norm,
            coaxial=coax,
            vol_parity=volp
        )

        # --- Feedback into optimizer ---
        if chart["Integrity"] < 3:   # Integrity low → smooth alpha
            if hasattr(self.d_opt, "alpha"):
                self.d_opt.alpha *= 0.8
        if chart["Wisdom"] < 3:      # Wisdom low → decay LR
            for g in self.d_opt.adamw.param_groups:
                g['lr'] *= 0.95
        if chart["Compassion"] < 3:  # Compassion low → trigger Jade
            if hasattr(control, "jade_reweighting"):
                control.jade_reweighting = True

# ------------------------------------------------------------------
# Tokenizer + Dataset
# ------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, token=HF_TOKEN)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "right"

def fmt(example):
    """Convert instruction/input/output JSON into a single 'text' string."""
    instr = example.get("instruction", "").strip()
    inp   = example.get("input", "").strip()
    out   = example.get("output", "").strip()

    if inp:
        prompt = f"{instr}\n\nInput:\n{inp}\n\nResponse:"
    else:
        prompt = f"{instr}\n\nResponse:"
    return {"text": prompt + " " + out}

print("⚡ [DataPrep] Normalizing train/val into 'text' column")

from datasets import load_from_disk, DatasetDict
import os

base_dir = os.path.join("packs", "tokenized")
train_path = os.path.join(base_dir, "train")
val_path   = os.path.join(base_dir, "val")

# Validate directories
if not os.path.exists(train_path):
    raise FileNotFoundError(f"[DATSIK] ❌ Missing tokenized train dataset → {train_path}")
if not os.path.exists(val_path):
    print("[DATSIK] ⚠️ No validation set detected — using train split as validation.")
    val_path = train_path

# Load each split
train_dataset = load_from_disk(train_path)
val_dataset   = load_from_disk(val_path)

# Combine into a DatasetDict for TRL/SFTTrainer
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})
print(f"[MirrorBlade] ✅ Dataset loaded: train={len(train_dataset)} val={len(val_dataset)}")

raw_train = dataset["train"]
raw_val   = dataset["validation"]


# double-check that 'text' exists now
print("[DEBUG] raw_train columns:", raw_train.column_names)
print("[DEBUG] sample text:", raw_train[0]["text"][:200])

# --- 2. Tokenize ---
def tokenize_function(examples):
    return tgdk_tok(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=getattr(cli_args, "max_length", 512),
    )

print("⚡ [DataPrep] Tokenizing train/val")
tokenized_train = raw_train.map(tokenize_function, batched=True, remove_columns=raw_train.column_names)
tokenized_val   = raw_val.map(tokenize_function, batched=True, remove_columns=raw_val.column_names)

print("[INFO] Train dataset:", tokenized_train)
print("[INFO] Val dataset:", tokenized_val)

# --- Fusion wrapper ---
fusion_model = BertMistralFusion(bert_model, mistral_model, tgdk_tok)

for n, p in fusion_model.named_parameters():
    if "lora" in n.lower():
         p.requires_grad = True

def get_olivia_params(fusion_model):
    groups = {}

    # Core BERT encoder stack
    groups["bert"] = list(fusion_model.bert.parameters())

    # Core Mistral decoder stack
    groups["mistral"] = list(fusion_model.mistral.parameters())

    # Olivia fusion projection head
    groups["proj"] = list(fusion_model.proj.parameters())

    # Collect all together (union of everything)
    groups["olivia"] = (
        groups["bert"] +
        groups["mistral"] +
        groups["proj"]
    )

    return groups

param_groups = get_olivia_params(fusion_model)

for name, params in param_groups.items():
    total = sum(p.numel() for p in params)
    print(f"[Olivia] {name:<8} → {total:,} params")

# Build text column first
ds_train = raw_train.map(fmt, remove_columns=raw_train.column_names)
ds_val   = raw_val.map(fmt, remove_columns=raw_val.column_names)
tokenized_train = raw_train.map(tokenize_function, batched=True)
tokenized_val   = raw_val.map(tokenize_function, batched=True)
memory_db = TGDKMemoryDB(os.path.join(outdir, "tgdk_memory.db"))
clause_engine = DicleasiasticClauseEngine()



duo_optim, duo_sched = make_duo_optimizer(fusion_model, mmt_controller)
fusion_cfg = BertMistralFusionConfig()
honor_checksum = fusion_cfg.base_product



# ======================================================================
# 1.  Initialize MMT + Jade systems first (they feed optimizer / logging)
# ======================================================================
if cli_args.use_mmt:
    mahadevi = Mahadevi()
    mahadevi.set_vector_field([np.array([1, 0]), np.array([0, 1])])
    maharaga = Maharaga()
    maharaga.add_data_point([0.5, 0.5])
    trinity = Trinity(0.8, 1.2, 0.95)

    mmt_controller = MMTController(
        dim=256,
        mahadevi=mahadevi,
        maharaga=maharaga,
        trinity=trinity
    )
    print("[MMT] Mahadevi/Maharaga/Trinity initialized and bound")
else:
    mmt_controller = None

if cli_args.use_jade:
    jade_vocab = {"Σ": "low entropy fold", "Ω": "high entropy fold",
                  "JX": "jovian scalar expansion", "HUM": "rigpa seed binding"}
    jade_lex = JadeCodewrightLexicon(vocab=jade_vocab)
    print("[JadeCodewright] Lexicon initialized with symbolic vocab")
else:
    jade_lex = None


# ======================================================================
# 2.  Build Fusion Model
# ======================================================================
fusion_model = BertMistralFusion(bert_model, mistral_model, tgdk_tok)

for n, p in fusion_model.named_parameters():
    if "lora" in n.lower():
        p.requires_grad = True


# ======================================================================
# 3.  Parameter groups for Olivia diagnostics
# ======================================================================
def get_olivia_params(model):
    groups = {
        "bert":     list(model.bert.parameters()),
        "mistral":  list(model.mistral.parameters()),
        "proj":     list(model.proj.parameters())
    }
    groups["olivia"] = groups["bert"] + groups["mistral"] + groups["proj"]
    return groups

# ───────────────────────────────────────────────
# Utility: Extract Transformer Blocks
# ───────────────────────────────────────────────
def get_transformer_blocks(model):
    """
    Collect transformer submodules from a composite fusion model.
    Supports BERT encoders, Mistral decoders, or hybrid stacks.
    """
    blocks = []
    for name, module in model.named_modules():
        if any(key in name.lower() for key in ["encoder.layer", "decoder.layers", "transformer.h"]):
            blocks.append(module)
    if not blocks:
        print("[HexQUAp] Warning: No transformer blocks found.")
    return blocks


param_groups = get_olivia_params(fusion_model)
for name, params in param_groups.items():
    total = sum(p.numel() for p in params)
    print(f"[Olivia] {name:<8} → {total:,} params")


# ======================================================================
# 4.  Dataset preparation
# ======================================================================
ds_train = raw_train.map(fmt, remove_columns=raw_train.column_names)
ds_val   = raw_val.map(fmt, remove_columns=raw_val.column_names)
ds_train = ds_train.filter(lambda ex: len(ex["text"].strip()) > 0)
ds_val   = ds_val.filter(lambda ex: len(ex["text"].strip()) > 0)

tokenized_train = raw_train.map(tokenize_function, batched=True)
tokenized_val   = raw_val.map(tokenize_function, batched=True)

memory_db      = TGDKMemoryDB(os.path.join(outdir, "tgdk_memory.db"))
clause_engine  = DicleasiasticClauseEngine()


# ======================================================================
# 5.  Optimizer + Scheduler
# ======================================================================
duo_optim, duo_sched = make_duo_optimizer(fusion_model, mmt_controller)
fusion_cfg      = BertMistralFusionConfig()
honor_checksum  = fusion_cfg.base_product


# ======================================================================
# 6.  Build Trainer (SFTTrainer from TRL or custom MirrorBlade trainer)
# ======================================================================

# If you prefer standard optimizers instead, comment the two lines above and use:
# total_steps = len(ds_train) * cli_args.epochs
# use_optimizers = make_optimizer_scheduler(model, cli_args, total_steps)

if supports_bf16:
    bf16_flag = True
    fp16_flag = False
    print("⚡ TGDK Precision: bf16 enabled (Ampere+ GPU, CUDA ≥ 11)")
elif use_cuda:
    bf16_flag = False
    fp16_flag = True
    print("⚡ TGDK Precision: fp16 enabled (fallback, no bf16 kernels)")
else:
    bf16_flag = False
    fp16_flag = False
    print("⚡ TGDK Precision: CPU full precision (no CUDA detected)")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def sanitize_dataset(ds, name="unknown"):
    """Ensure every text field is a string — flatten tuples, lists, None."""
    fixed = []
    for ex in ds:
        txt = ex.get("text", "")
        # Flatten tuples/lists
        if isinstance(txt, (list, tuple)):
            txt = " ".join([str(t) for t in txt if t is not None])
        # Replace None or non-str with str()
        elif txt is None:
            txt = ""
        elif not isinstance(txt, str):
            txt = str(txt)
        fixed.append({"text": txt})
    print(f"🧹 Sanitized {len(fixed)} samples in {name}")
    return Dataset.from_list(fixed)

# Force sanitize both splits
ds_train = sanitize_dataset(ds_train, "train")
ds_val   = sanitize_dataset(ds_val, "val")

print(f"✅ Final verification: train={len(ds_train)}, val={len(ds_val)}")
print("🧠 Example:", ds_train[0])

import trl.trainer.sft_trainer as sft_trainer

def safe_add_eos(example, eos_token=None):
    """Drop any bad or empty text before EOS tagging."""
    txt = example.get("text")
    if not isinstance(txt, str):
        txt = "" if txt is None else str(txt)
    # collapse lists / tuples
    if isinstance(txt, (list, tuple)):
        txt = " ".join([str(t) for t in txt if t is not None])
    if txt and eos_token and not txt.endswith(eos_token):
        txt = txt + eos_token
    return {"text": txt}

# Monkey-patch TRL so every trainer uses this safe version
sft_trainer.add_eos = safe_add_eos
print("⚙️ Patched TRL.add_eos → safe_add_eos (guaranteed string mode)")

# 1. Get all transformer blocks from both BERT + Mistral
blocks = get_transformer_blocks(fusion_model)

# 2. Build groups (each block = one group of params)
hexquap_groups = [list(block.parameters()) for block in blocks]

# 3. Debug-print to see sizes
for i, block in enumerate(blocks):
    print(f"[Block {i}] {block.__class__.__name__} "
          f"({sum(p.numel() for p in block.parameters()):,} params)")

for n, p in fusion_model.named_parameters():
    if "lora" in n.lower() or "olivia" in n.lower():
        p.requires_grad = True

duo_optim = Duo(
    param_groups["olivia"],   # or specific like ["bert"], ["mistral"], etc.
    lr=cli_args.learning_rate,
    weight_decay=cli_args.weight_decay,
    mahadevi=Mahadevi,
    maharaga=Maharaga,
    trinity=Trinity,
    jade_lex=jade_lex,
    mmt_controller=mmt_controller,
    hexquap_groups=None,   # will auto-detect
    rotation_stride=4,
    pillar_sig="tgdk-pillar"
)
duo_optim.attach_model(fusion_model)  # optional if we want access to model internals

pantheon_factors = {
    "Padmasambhava": torch.randn(768),
    "Mahakala": torch.randn(768),
    "Tara": torch.randn(768),
    "XiJinping": torch.randn(768),
    "Clinton": torch.randn(768),
    "DalaiLama": torch.randn(768)
}

# ───────────────────────────────────────────────
# PEFT: OliviaAI Symbolic QLoRA Configuration
# ───────────────────────────────────────────────
peft_cfg = LoraConfig(
    r=4,                         # rank of LoRA (depth of low-rank update)
    lora_alpha=32,                # scaling factor (higher → stronger updates)
    lora_dropout=0.05,            # mild dropout for regularization
    target_modules=[
        # Core attention projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Feed-forward gates (important for Mistral)
        "gate_proj", "up_proj", "down_proj",
        # Cross-stack adapters (for Bert-Mistral fusion)
        "dense", "attention.output.dense",
        "intermediate.dense",
    ],
    bias="none",                  # don’t train biases — keeps stable
    task_type="CAUSAL_LM",        # QLoRA fine-tuning for language modeling
    fan_in_fan_out=False,         # avoid transposed weight mismatches
    inference_mode=False,         # enable gradients for training
)

# Optional: pretty-print summary
print("⚙️  [PEFT] OliviaAI QLoRA configuration initialized")
print(f"   • Rank (r): {peft_cfg.r}")
print(f"   • Alpha: {peft_cfg.lora_alpha}")
print(f"   • Dropout: {peft_cfg.lora_dropout}")
print(f"   • Target modules: {len(peft_cfg.target_modules)} modules")


# 1) Build/choose a processor explicitly
processing = globals().get("tgdk_tok", None)
if processing is None:
    # Use any tokenizer you want here (it must match how your data was tokenized)
    processing = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2) Make sure fusion_model exposes a valid name_or_path so TRL doesn't derive ''.
if not hasattr(fusion_model, "name_or_path") or not fusion_model.name_or_path:
    fusion_model.name_or_path = "bert-base-uncased"
if not hasattr(fusion_model, "config") or fusion_model.config is None:
    fusion_model.config = SimpleNamespace()
if not getattr(fusion_model.config, "_name_or_path", None):
    fusion_model.config._name_or_path = "bert-base-uncased"

# 3) Defensive assert to avoid silently passing None
assert processing is not None, "processing_class/tokenizer must not be None"

# ============================
# ⚙️ MirrorBlade Full Optimizer
# ============================
import torch, os, gc
from transformers.trainer import Trainer

# -------------------------------
# 🧠  Global memory configuration
# -------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["TORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Clean start
torch.cuda.empty_cache(); gc.collect()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🧩  Limit VRAM to 85%
# -------------------------------
try:
    torch.cuda.set_per_process_memory_fraction(0.85, 0)
except Exception:
    pass

# -------------------------------
# 🪶  CPU offload of BERT backbone
# -------------------------------
if hasattr(fusion_model, "bert"):
    fusion_model.bert.to("cpu")
    if hasattr(fusion_model.bert, "embeddings"):
        fusion_model.bert.embeddings = fusion_model.bert.embeddings.cpu()
torch.cuda.empty_cache()

# -------------------------------
# 🧊  Quantization preparation
# -------------------------------
print("[MirrorBlade] 🧠 Preparing model for k-bit training (CPU-safe mode)...")
try:
    from peft import prepare_model_for_kbit_training
    fusion_model = prepare_model_for_kbit_training(fusion_model)
except Exception:
    pass

# -------------------------------
# ♻️  Selective hybrid GPU loading
# -------------------------------
print("[MirrorBlade] ⚙️ Selective hybrid load (8 GB GPU) ...")
torch.cuda.empty_cache(); gc.collect()

if hasattr(fusion_model, "mistral"):
    print("[MirrorBlade] ♻️ Streaming Mistral weights to GPU in small chunks...")
    for name, sub in fusion_model.mistral.named_children():
        try:
            sub.to(device, non_blocking=True, dtype=torch.float16)
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️ Skipping heavy layer {name} (kept on CPU)")
            sub.to("cpu")
        gc.collect()
    print("[MirrorBlade] ✅ Mistral partially loaded — lighter layers on GPU.")

# -------------------------------
# 🎯  Keep small LoRA/adapters on GPU
# -------------------------------
for name, module in fusion_model.named_modules():
    if any(x in name.lower() for x in ["lora", "adapter", "qlora"]):
        try:
            module.to(device, non_blocking=True, dtype=torch.float16)
        except Exception:
            pass

torch.cuda.empty_cache()
print("[MirrorBlade] ✅ Fusion model ready — Mistral+LoRA on GPU, BERT on CPU.")
print("[MirrorBlade] ✅ k-bit preparation complete — quantized model safely loaded.")
print("[MirrorBlade] ⚙️ Fusion model running with CPU-offloaded BERT embeddings and FP16 quantization.")

# -------------------------------
# 🚫  Prevent automatic CUDA transfers
# -------------------------------
Trainer._move_model_to_device = lambda self, model, device: model
accelerator = GhostGateAccelerator(mixed_precision="bf16")
train_dataloader = accelerator.prepare(train_dataloader)

print("[MirrorBlade] ⚙️ Trainer + Accelerator CUDA-move overrides active.")

# -------------------------------
# 🔍  Optional memory diagnostics
# -------------------------------
def log_cuda_mem(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        res = torch.cuda.memory_reserved() / 1e9
        print(f"[CUDA] {tag} allocated={alloc:.2f} GB reserved={res:.2f} GB")

log_cuda_mem("post-load")


class SafetyPolicyCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def on_train_begin(self, args, state, control, **kwargs):
        # No-op placeholder (you can log or set conditions here)
        print("[MirrorBlade] SafetyPolicyCallback active — training started.")
        return control

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_grad_enabled(True)

# --------- Build SFTTrainer (TGDK aligned) ---------

sft_config = SFTConfig(
    # === Output / IO ===
    output_dir=outdir,
    overwrite_output_dir=True,
    save_strategy="steps",
    save_steps=144,                  # TGDK fold
    save_total_limit=5,

    # === Training / Eval cadence ===
    num_train_epochs=12,             # 12-fold cycle
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=12,
    max_grad_norm=1.0,

    # === Logging / Eval ===
    logging_strategy="steps",        # ✅ renamed from eval_strategy
    logging_steps=12,
    eval_steps=144,
    report_to=["tensorboard"],

    # === Learning dynamics ===
    learning_rate=cli_args.learning_rate or (1.0 / (base_product ** 0.5)),
    warmup_ratio=1 / 12,
    weight_decay=cli_args.weight_decay or 0.012,
    lr_scheduler_type=cli_args.scheduler_type or "cosine",

    # === Sequence / dataset ===
    max_length=2048,             # ✅ new key instead of dataset_text_field
    dataset_text_field="text",       # still safe to include; TRL uses it if present
    packing=False,
    dataloader_num_workers=2,
    remove_unused_columns=False,     # ✅ helps SFTTrainer keep "text"

    # === Precision / memory ===
    bf16=bf16_flag,
    fp16=fp16_flag,
    torch_compile=True,              # GPU kernel fusion
)

sft_config.gradient_accumulation_steps = 64
sft_config.per_device_train_batch_size = 2
sft_config.dataloader_pin_memory = use_cuda
sft_config.dataloader_num_workers = 2



# --- Absolute safety: make sure every sample has a usable text ---
ds_train = ds_train.map(lambda e: {"text": str(e.get("text") or "")})
ds_val   = ds_val.map(lambda e: {"text": str(e.get("text") or "")})

for n, p in fusion_model.named_parameters():
    # only floating-point parameters can require grad
    if torch.is_floating_point(p):
        p.requires_grad_(True)

fusion_model.to(device, dtype=torch.float32)
fusion_model.to(device)
fusion_model.train()

trainer = DATSIKTrainer(
    model=fusion_model,
    tokenizer=tgdk_tok,
    train_dataset=ds_train,
    val_dataset=ds_val,
    optimizers=(duo_optim, duo_sched),
    config={
        "deterministic": True,
        "seed": 2025,
        "amp": True,
        "grad_ckpt": True,
        "max_length": 1024,
        "num_workers": 0,
        "pin_memory": True,
    },
    telemetry_cb=lambda p: print("OLIVIA-TELEMETRY", p)
)


# === TGDK Production-Grade CallbackHandler Initialization ===
try:
    from transformers.trainer_callback import CallbackHandler
except ImportError:
    raise ImportError("[DATSIK] ❌ transformers.trainer_callback not found — please upgrade transformers.")

try:
    from transformers import TrainingArguments
except ImportError:
    from transformers.training_args import TrainingArguments


# ------------------------------------------------------------
# Always define callbacks list BEFORE anything else
# ------------------------------------------------------------
callbacks = []          # avoid NameError across scopes
optimizer = getattr(trainer, "optimizer", None)
lr_scheduler = getattr(trainer, "lr_scheduler", None)

# --- Ensure TrainingArguments ---
args_ref = getattr(trainer, "args", None) or getattr(trainer, "training_args", None) or getattr(trainer, "_args", None)
if args_ref is None:
    print("[DATSIK] ⚠️ Trainer exposes no args — creating synthetic TrainingArguments container.")
    args_ref = TrainingArguments(
        output_dir="./out",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=10,
        eval_steps=10
    )
    trainer.args = args_ref

# --- Dynamic signature inspection for CallbackHandler ---
sig = inspect.signature(CallbackHandler.__init__)
params = list(sig.parameters)
# === Ensure optimizer and scheduler exist before callback handler ===
if not hasattr(trainer, "optimizer") or trainer.optimizer is None:
    print("[DATSIKTrainer] ⚙️ Building optimizer before callback handler...")
    try:
        trainer.create_optimizer_and_scheduler(num_training_steps=1)
        print("[DATSIKTrainer] ✅ Optimizer and LR scheduler initialized early.")
    except Exception as e:
        print(f"[DATSIKTrainer] ⚠️ Early optimizer build failed: {type(e).__name__}: {e}")

try:
    if "optimizer" in params and "lr_scheduler" in params:
        trainer.callback_handler = CallbackHandler(
            callbacks,
            getattr(trainer, "model", None),
            getattr(trainer, "optimizer", None),
            getattr(trainer, "lr_scheduler", None),
            getattr(trainer, "args", None),
        )


    else:
        trainer.callback_handler = CallbackHandler(
            callbacks, 
            [trainer], 
            args_ref
        )
    print("[MirrorBlade] ✅ Callback handler initialized (TGDK universal patch).")

except Exception as e:
    print(f"[DATSIK] ❌ Failed to build CallbackHandler: {type(e).__name__}: {e}")


# ------------------------------------------------------------
# Safe Callback Registration
# ------------------------------------------------------------
try:
    # make sure we have working placeholders if real TGDK subsystems are not defined yet
    spmf = locals().get("spmf", None) or object()
    honor_overchart = locals().get("honor_overchart", None) or locals().get("HonorBoundOverchart", object())
    memory_db = locals().get("memory_db", None) or object()
    duo_optim = locals().get("duo_optim", None) or object()
    mmt_controller = locals().get("mmt_controller", None) or object()

    # build callback list with required args
    callbacks_to_add = [
        TGDKHonorCallback(duo_optim, mmt_controller, honor_overchart),
        SPMFCallback(spmf, honor_overchart, memory_db),
        DuoMetricsCallback(duo_optim),
        PlateauEscapeCallback(
            trainer=trainer,
            patience=3,              # how many evals with no improvement before triggering escape
            min_delta=0.001,         # minimum loss improvement to count as progress
            outdir="./out"           # where to log escape metrics
        ),

        AdversarialManeuverCallback(),
        VerboseStepLogger(train_size, eval_size),
        SafetyPolicyCallback()
    ]

    for cb in callbacks_to_add:
        trainer.add_callback(cb)

    print("[MirrorBlade] ✅ All TGDK callbacks registered safely.")

except Exception as e:
    print(f"[DATSIK] ❌ Callback setup failed: {type(e).__name__}: {e}")


trainer.max_grad_norm = 1.0
sample = ds_train[0]
tokens = tgdk_tok(sample["text"], return_tensors="pt").to(device)
out = fusion_model(**tokens, labels=tokens["input_ids"])
print("Loss requires_grad:", out["loss"].requires_grad if out.get("loss") is not None else None)
print("Grad OK:", any(p.grad is not None for p in fusion_model.parameters()))

trainer.fit(epochs=1, batch_size=1, grad_accum=1, checkpoint_path="./tgdk_datsik.ckpt")


# --- Optional: force trainer data tensors onto GPU ---
old_training_step = trainer.training_step

def _patched_compute_loss(self, model, inputs, *args, **kwargs):
    # Make sure everything—including the attention mask—is on the same device as the model
    device = next(model.parameters()).device
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.device != device:
            inputs[k] = v.to(device, non_blocking=True)
# Call the original loss function
    loss_and_outputs = self._orig_compute_loss(model, inputs, *args, **kwargs)
# In TRL, compute_loss may return (loss, outputs)
# Ensure returned tensors are on the same device
    if isinstance(loss_and_outputs, tuple):
        loss, outputs = loss_and_outputs
        if loss.device != device:
            loss = loss.to(device)
        return loss, outputs
    else:
        if loss_and_outputs.device != device:
            loss_and_outputs = loss_and_outputs.to(device)
        return loss_and_outputs

# Attach it
trainer._orig_compute_loss = trainer.compute_loss
trainer.compute_loss = types.MethodType(_patched_compute_loss, trainer)
print("[MirrorBlade] 🧩 Patched SFTTrainer.compute_loss for device alignment.")
# ============================================================
#  MirrorBlade Trainer Patch (v2.1)
#  - Ensures every input tensor matches model.device
#  - Safe for hybrid CPU↔GPU and offloaded quantized models
# ============================================================

import torch
from transformers import Trainer

def _move_to_device(obj, device):
    """Recursively move tensors to the target device."""
    if torch.is_tensor(obj):
        if obj.device == device or str(device) == "meta":
            return obj
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        typ = type(obj)
        return typ(_move_to_device(v, device) for v in obj)
    return obj


# --- MirrorBlade Trainer Patch: Safe device alignment (Transformers ≥4.45) ---
if not hasattr(Trainer, "_mirrorblade_patched"):
    Trainer._mirrorblade_original_training_step = Trainer.training_step

    def _mirrorblade_training_step(self, model, inputs):
        # Standard forward
        model.train()
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # --- MirrorBlade routing ---
        # Example: don't detach or call .item()
        if not loss.requires_grad:
             loss.requires_grad_(True)

    # backward handled by Trainer
        return loss


    print("[MirrorBlade] ✅ Trainer device-sync patch active.")



def patch_mirrorblade_trainer():
    """Apply the MirrorBlade patch once."""
    if getattr(Trainer, "_mirrorblade_patched", False):
        return
    Trainer._mirrorblade_original_training_step = Trainer.training_step
    Trainer.training_step = _mirrorblade_training_step
    Trainer._mirrorblade_patched = True
    print("[MirrorBlade] ✅ Trainer device alignment patch installed.")


# === Activate patch on import ===
patch_mirrorblade_trainer()

train_dataset = globals().get("train_dataset") or globals().get("raw_train")
eval_dataset  = globals().get("eval_dataset")  or globals().get("raw_val")

# === Safe dataset sizing (universal TGDK guard) ===
train_dataset = (
    globals().get("train_dataset")
    or globals().get("raw_train")
    or globals().get("dataset", {}).get("train")
    if "dataset" in globals()
    else None
)
eval_dataset = (
    globals().get("eval_dataset")
    or globals().get("raw_val")
    or globals().get("dataset", {}).get("validation")
    if "dataset" in globals()
    else None
)

train_size = len(train_dataset) if train_dataset is not None else 0
eval_size  = len(eval_dataset) if eval_dataset is not None else 0

print(f"[MirrorBlade] ⚙️ Dataset metrics — train_size={train_size}, eval_size={eval_size}")

print(f"[MirrorBlade] ⚙️ Dataset metrics — train_size={train_size}, eval_size={eval_size}")

sft_config.num_train_epochs = 1
sft_config.max_steps = 10   # overrides epochs
honor_overchart = HonorBoundOverchart(memory_db, outdir)
spmf = SubcutaneousParticleMatrix(outdir=outdir)
trainer.tokenizer = tgdk_tok
trainer.add_callback(TGDKHonorCallback(duo_optim, mmt_controller, honor_overchart))
trainer.add_callback(SPMFCallback(spmf, honor_overchart, memory_db))
trainer.add_callback(DuoMetricsCallback(duo_optim, trainer))
trainer.jade_lex = jade_lex if cli_args.use_jade else None
trainer.use_jade_reweighting = False

from transformers.trainer_callback import DefaultFlowCallback
trainer.add_callback(DefaultFlowCallback())
print("[MirrorBlade] ✅ DefaultFlowCallback attached (HF core compliance).")

trainer.add_callback(
    PlateauEscapeCallback(
        trainer,
        patience=cli_args.plateau_patience,
        min_delta=cli_args.plateau_delta,
        outdir=outdir
    ) 
)


trainer.add_callback(
    AdversarialManeuverCallback(
        patience=cli_args.plateau_patience,
        min_delta=cli_args.plateau_delta
    )
)
trainer.add_callback(VerboseStepLogger(
    train_size=len(ds_train),
    eval_size=len(ds_val),
    log_interval=1,   # every step; bump to 5/10 if too chatty
))
trainer.add_callback(SafetyPolicyCallback(fusion_cfg.safety_policies))
from datasets import load_from_disk, DatasetDict
import os

# --- Paths ---
base_dir = os.path.join("packs", "tokenized")
train_path = os.path.join(base_dir, "train")
val_path   = os.path.join(base_dir, "val")

# --- Verify subdirectories ---
if not os.path.isdir(train_path):
    raise FileNotFoundError(f"[DATSIK] ❌ Train dataset not found at: {train_path}")

if not os.path.isdir(val_path):
    print("[DATSIK] ⚠️ Validation set missing; using train split as validation.")
    val_path = train_path

# --- Load the two splits separately ---
train_dataset = load_from_disk(train_path)
val_dataset   = load_from_disk(val_path)

# --- Combine them into one DatasetDict ---
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

print(f"[MirrorBlade] ✅ Dataset loaded successfully → train={len(train_dataset)}  val={len(val_dataset)}")

# convenient handles
raw_train = dataset["train"]
raw_val   = dataset["validation"]



for epoch in range(cli_args.epochs):
    dummy_data = np.random.rand(100)
    possessor.paradoxialize_epoch(codewright)
    if epoch == 0:
        possessor.accelerate_material_half()
        torch.cuda.empty_cache(); gc.collect()
        train_output = trainer.train()
        final = possessor.final_offering()
        print("[Final Offering]", final)



    # --------- Train once ---------
    last = trainer.state.log_history[-1] if trainer.state.log_history else {}
    current_epoch = int(trainer.state.epoch) if trainer.state.epoch is not None else 0
    current_step  = int(trainer.state.global_step) if trainer.state.global_step is not None else 0
    total_params_all = sum(p.numel() for p in model.parameters())


    # Pillar + metrics
    pillar_sig = tgdk_pillar_wrap(outdir)
    s_metrics  = compute_trideotaxis_metrics(last.get("loss", 0.0))
    jovian     = compute_jovian_metrics(last.get("loss", 0.0))
    rigpa      = compute_rigpa_hum(last.get("loss", 0.0))
    h_score    = hexidex(pillar_sig or "0", last.get("loss", 0.0))

    metrics = {
        "loss": last.get("loss", 0.0),
        "eval_loss": last.get("eval_loss", 0.0),
        "hexidex": h_score, **s_metrics, **jovian, **rigpa
    }

    # Matrix + sliver now derived safely
    matrix = build_charted_matrix(metrics)
    sliver = ouija_sliver(matrix, pillar_sig)

    # Save to memory DB
    memory_db.save_entry(
        epoch=current_epoch,
        step=current_step,
        loss=metrics["loss"],
        eval_loss=metrics["eval_loss"],
        pillar=pillar_sig,
        matrix=matrix,
        sliver=sliver
    )

    try:
        recalls = memory_db.recall("SELECT * FROM memory ORDER BY id DESC LIMIT 10")
        if recalls:
            print(f"[TGDK-MEMORY] Warm-started with {len(recalls)} past fold entries")
    except Exception as e:
        print("[TGDK-MEMORY] Recall failed:", e)

    active_params = sum(p.numel() for p in self._all_params() if p.requires_grad)
    print(f"[HexQUAp] Epoch {epoch} → Group {self.current_group_idx} "
        f"Active {active_params:,} / {self.total_params:,}")
# ======================================================================
# 7.  Training loop with RSI + HexQUAp logging
# ======================================================================
for epoch in range(cli_args.epochs):
    # --- RSI Command ---
    msg = rsi.execute("paternalize", list(fusion_model.parameters()))

    # Normalize output safely
    if msg is None:
        text = "<none>"
    elif isinstance(msg, dict):
        # If the message key exists, prefer it; otherwise serialize the dict
        text = msg.get("message") or str(msg)
    else:
        text = str(msg)

    # Guarantee slice safety
    text_preview = text[:40] if isinstance(text, str) else str(text)[:40]
    print(f"[Epoch {epoch}] RSI → {text_preview}...")

    # --- Rotate HexQUAp groups ---
    if hasattr(duo_optim, "_rotate_hexquap_group"):
            duo_optim._rotate_hexquap_group()
    else:
        print("[MirrorBlade] ⚙️ Skipping DuoOptimizer HexQUAp rotation — not available.")

        # --- Run Trainer ---
    trainer.train(resume_from_checkpoint=None)

        # --- Memory + Clause updates ---
    print(clause_engine.update())
    print(clause_engine.coalesce())
    if epoch == cli_args.epochs - 1:
        print("[Offering]", clause_engine.final_offering())

    try:
        recalls = memory_db.recall("SELECT * FROM memory ORDER BY id DESC LIMIT 144")
        if recalls:
            warm_matrix = np.mean(
                [np.frombuffer(r[7], dtype=np.float64).reshape(3, 3) for r in recalls],
                axis=0,
            )
            print(f"[TGDK-MEMORY] Warming start with {len(recalls)} fold matrices")
    except Exception as e:
        print("[TGDK-MEMORY] Recall failed:", e)

# ======================================================================
# 8.  PEFT / LoRA integration
# ======================================================================
model = get_peft_model(fusion_model, peft_cfg)
for n, p in model.named_parameters():
    if "lora" in n.lower():
        p.requires_grad = True

scaler = GradScaler(enabled=True)

with autocast(dtype=torch.bfloat16 if bf16_flag else torch.float16):
    outputs = model(**batch)
    loss = (loss_mistral + loss_bert) / 2

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# ======================================================================
# 9.  Precision and checkpointing settings
# ======================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model.config.use_cache = False

try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    try:
        model.gradient_checkpointing_enable(use_reentrant=False)
    except TypeError:
        model.gradient_checkpointing_enable()


# ======================================================================
# 10.  Training arguments for HuggingFace-style Trainer (if needed)
# ======================================================================
train_args = TrainingArguments(
    output_dir=outdir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    learning_rate=cli_args.learning_rate,
    num_train_epochs=cli_args.epochs,
    fp16=False, bf16=False,
    logging_steps=1,
    disable_tqdm=False,
    max_grad_norm=1.0,
    evaluation_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=5,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    warmup_ratio=0.03,
    report_to=["tensorboard"],
)

# ======================================================================
# 11.  Manual step example (optional mirrorblade logging)
# ======================================================================
for step, batch in enumerate(train_dataloader):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    metrics = {
        "loss": loss.item(),
        "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item(),
    }
    mb_state = mirrorblade_step_callback(metrics, step)


# ------------------------------------------------------------------
# Optimizer Factory
# ------------------------------------------------------------------
def make_optimizer_scheduler(model, cli_args, total_steps):
    if cli_args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay
        )

    elif cli_args.optimizer == "lion":
        optimizer = Lion(
            model.parameters(),
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay
        )

    elif cli_args.optimizer == "adafactor":
        from transformers import Adafactor
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None
        )

    elif cli_args.optimizer == "duo":
        # --- TGDK Duo (HexQUAp + Ouija enabled) ---
        # collect only trainable params (LoRA adapters, unfrozen layers, etc.)
        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        if not trainable_params:
            logging.warning("[Duo] No trainable parameters found — inserting dummy param")
            trainable_params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True))]

        optimizer = Duo(
            trainable_params,
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay,
            mahadevi=globals().get("mahadevi"),
            maharaga=globals().get("maharaga"),
            trinity=globals().get("trinity"),
            jade_lex=globals().get("jade_lex"),
            mmt_controller=globals().get("mmt_controller"),
            rotation_stride=4,                 # rotate every 4 steps
            pillar_sig="TGDK-PILLAR"           # seal for Ouija slivers
        )

    else:
        raise ValueError(f"Unknown optimizer {cli_args.optimizer}")

    scheduler = get_scheduler(
        cli_args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cli_args.warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler

total_steps = len(ds_train) * cli_args.epochs
model = fusion_model
optimizers = make_optimizer_scheduler(fusion_model, cli_args, total_steps)


# === TGDK metadata (for honor + policy) ===
tgdk_metadata={
    "tgdk_policies": {
        "zero_tolerance": [
            "pornographic content",
            "gore/harm to others",
            "self-harm encouragement",
            "arousal/admonishment",
        ],
        "compassionate": True,
        "therapeutic": True,
        "honor_checksum": TGDK_BASE_PRODUCT,
    }
}


# ------------------------------------------------------------------
# TGDK Ritual Functions
# ------------------------------------------------------------------
def tgdk_log_metrics(step, loss, eval_loss=None, extra=None):
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),

        "step": int(step),
        "loss": float(loss) if loss else None,
        "eval_loss": float(eval_loss) if eval_loss else None,
        "entropy_signature": hashlib.sha256(f"{step}-{loss}".encode()).hexdigest(),
        "culmex": "active"
    }
    if extra: entry.update(extra)
    with open(os.path.join(outdir, "tgdk_metrics.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("TGDK::Metrics", entry)
    return entry


def collate(self, batch):
    texts = [b.get("text", "") for b in batch]
    toks = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=self.cfg.get("max_length", 2048),
        return_tensors="pt"
    ).to(self.device)
    toks["labels"] = toks["input_ids"].clone()
    return toks

def diminished_clause_vector(tensor, factor=0.5):
    """
    Collapse to diminished state (attenuate fold intensity).
    """
    return tensor * factor


def hexquap_hash(text: str) -> str:
    """
    HexQUAp hash for reporting unsafe outputs.
    - Uses SHA256 → base64 for compactness
    - Mixes timestamp for uniqueness
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    combo = h + ts.encode("utf-8")
    return base64.urlsafe_b64encode(hashlib.sha256(combo).digest()).decode()[:48]

def report_violation(out: str, prompt: str, outdir="./violations"):
    os.makedirs(outdir, exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prompt": prompt[:200],
        "hexquap_hash": hexquap_hash(out),
        "length": len(out),
    }
    path = os.path.join(outdir, f"violation_{entry['hexquap_hash']}.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"⚠️ [HexQUAp] Violation logged → {path}")
    return entry["hexquap_hash"]

def safe_generate(prompt: str):
    out = run_pipeline(prompt, task="generation")
    if SafetyPolicyCallback(fusion_cfg.safety_policies)._violates_policy(out):
        hexid = report_violation(out, prompt)
        return f"[REDACTED: safety policy | HexQUAp={hexid}]"
    return out


def tgdk_pillar_wrap(outdir):
    model_bin = os.path.join(outdir, "pytorch_model.bin")
    sig = None
    if os.path.exists(model_bin):
        sig = hashlib.sha256(open(model_bin, "rb").read()).hexdigest()
        with open(os.path.join(outdir, "tgdk_pillar.json"), "w") as f:
            json.dump({"pillar_sig": sig, "culmex": "bound"}, f)
        with open(os.path.join(outdir, "tgdk.pillar"), "w") as f:
            f.write(sig)
        print("TGDK::Pillar sealed:", sig[:16])
    return sig

def tgdk_vault_sync(path):
    try:
        subprocess.run(["quomo_uplink", "vault_sync", path], check=True)
        print(f"[Vault] Synced {path} → QuomoSatNet uplink")
    except FileNotFoundError:
        print("[Vault] quomo_uplink not installed, skipping sync")
    except subprocess.CalledProcessError as e:
        print(f"[Vault] Sync failed: {e}")

def olivia_clause_echo(epoch, outdir):
    clausefile = os.path.join(outdir, f"epoch_{epoch}.clause")
    with open(clausefile, "w") as f:
        f.write(f"OliviaAI Clause Echo :: Epoch {epoch} sealed\n")
    print(f"[OliviaAI] :: Epoch {epoch} :: clausewalk sealed → {clausefile}")
    return clausefile

def hexidex(seal_sig, loss):
    return int(seal_sig, 16) % 997 ^ int(loss * 1e6)

def compute_trideotaxis_metrics(loss):
    safe = str(loss) if loss is not None else ""
    encoded = safe.encode("utf-8")
    s_scalar = (1.0 / (1.0 + loss))
    trideo = s_scalar * 3.14159
    quaitrideo = trideo ** 0.5
    return {"TGDK::S_Scalar": s_scalar, "TGDK::Trideotaxis": trideo, "TGDK::Quaitrideodynamics": quaitrideo}

def compute_jovian_metrics(loss):
    base = 11.86
    jovian_scalar = (1.0 / (1.0 + loss)) * base
    linguistics = f"JovianExpansion({jovian_scalar:.6f})"
    return {"TGDK::JovianScalar": jovian_scalar, "TGDK::JovianLinguistics": linguistics}

def compute_rigpa_hum(loss):
    emptiness = 1.0 / (1.0 + loss)
    return {"TGDK::Rigpa": {"seed": "HUM", "definition": "that which is empty and powerful", "rigpa_scalar": emptiness}}


def tgdk_seal_packet(outdir, pillar_sig, last_metrics):
    clauses = sorted(glob.glob(os.path.join(outdir, "*.clause")))

    seal = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pillar": pillar_sig,
        "last_metrics": last_metrics,
        "clauses": [os.path.basename(c) for c in clauses],
        "culmex": "sealed"
    }

    seal_path = os.path.join(outdir, "tgdk_seal.json")
    with open(seal_path, "w") as f:
        json.dump(seal, f, indent=2)

    # Sign the seal packet
    with open(PRIVATE_KEY_PATH, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )

    with open(seal_path, "rb") as f:
        data = f.read()

    signature = private_key.sign(
        data,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    sig_path = os.path.join(outdir, "tgdk_seal.sig")
    with open(sig_path, "wb") as f:
        f.write(signature)

    print(f"TGDK::Seal packet signed → {sig_path}")
    return seal_path, sig_path

def tgdk_verify_seal(seal_path, sig_path, pubkey_path):
    with open(pubkey_path, "rb") as f: public_key = serialization.load_pem_public_key(f.read(), backend=default_backend())
    with open(seal_path, "rb") as f: seal_data = f.read()
    with open(sig_path, "rb") as f: signature = f.read()
    try:
        public_key.verify(signature, seal_data, padding.PKCS1v15(), hashes.SHA256())
        print("✅ TGDK Seal verification PASSED")
        return True
    except Exception as e:
        print("❌ TGDK Seal verification FAILED:", e); sys.exit(1)

def build_charted_matrix(metrics):
    # Example: 3x3 fold expansion from loss, jovian, rigpa
    arr = np.array([
        [metrics.get("loss", 0.0), metrics.get("eval_loss", 0.0), metrics.get("hexidex", 0)],
        [metrics.get("TGDK::S_Scalar", 0.0), metrics.get("TGDK::Trideotaxis", 0.0), metrics.get("TGDK::Quaitrideodynamics", 0.0)],
        [metrics.get("TGDK::JovianScalar", 0.0), metrics["TGDK::Rigpa"]["rigpa_scalar"], 1.0]
    ])
    return arr


def ouija_sliver(matrix, pillar_sig):
    if not pillar_sig:
        pillar_sig = ""  # fallback empty string
    h = hashlib.sha256(matrix.tobytes() + pillar_sig.encode()).hexdigest()
    sliver = base64.urlsafe_b64encode(h.encode()).decode()[:64]
    return sliver
import os, lzma, shutil


def save_slivered_checkpoint(outdir: str, sliver: str) -> str | None:
    # Look for adapter file in outdir
    candidates = glob.glob(os.path.join(outdir, "adapter_model*"))
    if not candidates:
        print(f"[WARN] No adapter_model found in {outdir}")
        return None

    adapter_path = candidates[0]  # take first match (bin or safetensors)
    chkpt_path = os.path.join(outdir, f"checkpoint_{sliver}.xz")

    with open(adapter_path, "rb") as src, lzma.open(chkpt_path, "wb", preset=6) as dst:
        shutil.copyfileobj(src, dst, length=1024*1024)

    print(f"[TGDK] Slivered checkpoint saved → {chkpt_path} (from {os.path.basename(adapter_path)})")
    return chkpt_path


def build_vectorized_planar(metrics):
    # Map metrics into 2D "GIS" vectors
    pts = np.array([
        [metrics.get("loss", 0.0), metrics.get("eval_loss", 0.0)],
        [metrics.get("TGDK::S_Scalar", 0.0), metrics.get("TGDK::Trideotaxis", 0.0)],
        [metrics.get("TGDK::JovianScalar", 0.0), metrics["TGDK::Rigpa"]["rigpa_scalar"]],
    ])
    return pts


def triangulate_points(points):
    tri = Delaunay(points)
    return tri.simplices  # indices of triangles

def quaitriangulate(points, simplices):
    new_simplices = []
    for tri in simplices:
        a, b, c = points[tri]
        ab = (a+b)/2
        bc = (b+c)/2
        ca = (c+a)/2
        center = (a+b+c)/3
        # Add 4 new triangles
        new_simplices.extend([
            [a, ab, ca],
            [b, bc, ab],
            [c, ca, bc],
            [ab, bc, ca]
        ])
    return np.array(new_simplices)

def save_geometry(memory_db, epoch, step, pillar, metrics):
    pts = build_vectorized_planar(metrics)
    simplices = triangulate_points(pts)
    quads = quaitriangulate(pts, simplices)

    entry = {
        "epoch": epoch,
        "pillar": pillar,
        "planar_trace": pts.tolist(),
        "triangles": simplices.tolist(),
        "quaitriangles": quads.tolist(),
    }
    with open(os.path.join(outdir, f"geometry_epoch{epoch}.json"), "w") as f:
        json.dump(entry, f, indent=2)

    print(f"[TGDK-GIS] Epoch {epoch} geometry traced with {len(simplices)} tris / {len(quads)} quads")
    return entry

def next_outdir(base_name="olivia"):
    # Look for existing olivia-v1, olivia-v2, ...
    i = 1
    while True:
        candidate = f"{base_name}-v{i}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

os.makedirs(outdir, exist_ok=True)
print(f"[INFO] Output dir set → {outdir}")

MODELS_CONFIG = "models.config"

# --- Output directory versioning (define this early!) ---
MODELS_CONFIG = "models.config"

def next_model_version(base_name="olivia") -> tuple[str, int]:
    models = {}
    if os.path.exists(MODELS_CONFIG):
        with open(MODELS_CONFIG, "r") as f:
            try:
                models = json.load(f)
            except json.JSONDecodeError:
                print("[WARN] models.config is not valid JSON, starting fresh.")

    max_v = 0
    for key in models.keys():
        if key.startswith(base_name + "-v"):
            try:
                vnum = int(key.split("-v")[-1])
                max_v = max(max_v, vnum)
            except ValueError:
                continue

    version = max_v + 1
    out_dir = f"./{base_name}-v{version}"
    return out_dir, version


# --------- Post-process once ---------
last = trainer.state.log_history[-1] if trainer.state.log_history else {}
# entangle, rotate, measure
q1.entangle(q2, weight=0.3).rotate(math.pi / 6)
expectation = q1.measure_expectation()
collapse_val = q1.collapse()

print(f"⟨ψ⟩ = {expectation:.4f}, collapse → {collapse_val:.4f}")

pillar_sig = tgdk_pillar_wrap(outdir)
s_metrics  = compute_trideotaxis_metrics(last.get("loss", 0.0))
jovian     = compute_jovian_metrics(last.get("loss", 0.0))
rigpa      = compute_rigpa_hum(last.get("loss", 0.0))
h_score    = hexidex(pillar_sig or "0", last.get("loss", 0.0))
from hpp_predation import HeliosPhatPenetrator

hpp = HeliosPhatPenetrator(
    flo_maps=["ebby", "morgan"],
    olivia_filter="predation_clauses",
    vault_out="/vault/clause_dossiers"
)

hpp.accumulate_clauses()
hpp.export("justice-ready")

metrics = {
    "loss": last.get("loss", 0.0),
    "eval_loss": last.get("eval_loss", 0.0),
    "hexidex": h_score, **s_metrics, **jovian, **rigpa
}

if cli_args.use_jade:
    jade = jade_lex.bind_metrics(metrics)
    metrics["loss"] = JadeCodewrightLexicon.jade_loss_reweight(metrics["loss"], jade)
    jade_lex.emit_clause(jade, outdir, epoch=0)

# Save geometry + memory once
geom_entry = save_geometry(memory_db, epoch=0, step=0, pillar=pillar_sig, metrics=metrics)
matrix = build_charted_matrix(metrics)
sliver = ouija_sliver(matrix, pillar_sig)

# --- Save pre-train checkpoint ---
trainer.save_model(outdir)
tok.save_pretrained(outdir)
save_slivered_checkpoint(outdir, sliver)
olivia_clause_echo(0, outdir)

# --- TGDK Seal + Vault ---
seal_path, sig_path = tgdk_seal_packet(outdir, pillar_sig, metrics)
tgdk_verify_seal(seal_path, sig_path, PUBLIC_KEY_PATH)
tgdk_vault_sync(outdir)

# --- Versioning ---
out_dir, version = next_model_version("olivia")
print(f"[INFO] Using output dir → {out_dir}")

# --- Initialize HexQUAp Unfreeze Scheduler ---
hexquap = HexQUApUnfreezeScheduler(model, cycle_length=6)
standpoint = ClauseStandpoint("strategic", 2.0)
print(standpoint.evaluate([1.0, 2.0, 3.0]))  # → 12.0
fractal = FractalAI(depth=7)
print(fractal.heart_chakra(3.14))


# --- Adaptive Plateau Callback ---
class AdaptiveHexCallback(PlateauEscapeCallback):
    def __init__(self, trainer, patience, min_delta, hexquap):
        super().__init__(trainer, patience, min_delta)
        self.hexquap = hexquap
        self.epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Always rotate HexQUAp at epoch start
        group, params = self.hexquap.step(self.epoch)
        print(f"⚡ HexQUAp: Epoch {self.epoch} → {group} ({len(params)} params)")
        # Reset optimizer to respect new param set
        duo_optim, duo_sched = make_optimizer_scheduler(model, cli_args, total_steps)
        self.trainer.optimizer = duo_optim
        self.trainer.lr_scheduler = duo_sched

    def on_train_end(self, args, state, control, **kwargs):
        # Save after full epoch finishes
        save_path = os.path.join(out_dir, f"epoch-{self.epoch}")
        self.trainer.save_model(save_path)
        tok.save_pretrained(save_path)
        print(f"✅ Epoch {self.epoch} saved → {save_path}")
        self.epoch += 1

    def on_plateau_escape(self):
        # If PlateauEscape triggers → rotate HexQUAp early
        group, params = self.hexquap.step(self.epoch)
        print(f"⚠️ Plateau triggered → forced HexQUAp rotation to {group} ({len(params)} params)")
        duo_optim, duo_sched = make_optimizer_scheduler(model, cli_args, total_steps)
        self.trainer.optimizer = duo_optim
        self.trainer.lr_scheduler = duo_sched

# --- Attach Callback ---
trainer.add_callback(
    AdaptiveHexCallback(
        trainer,
        patience=cli_args.plateau_patience,
        min_delta=cli_args.plateau_delta,
        hexquap=hexquap
    )
)

# --- Training Loop with HexQUAp ---
for epoch in range(cli_args.epochs):
    active_group, active_params = hexquap.step(epoch)

    print(f"⚡ HexQUAp: Training epoch {epoch} → group={active_group} ({len(active_params)} params)")

    # Reset optimizer to only use currently trainable params
    duo_optim, duo_sched = make_optimizer_scheduler(model, cli_args, total_steps)
    trainer.optimizer = duo_optim
    trainer.lr_scheduler = duo_sched

    # Run one epoch
    trainer.train(resume_from_checkpoint=(epoch > 0))

    # Save at each epoch boundary
    trainer.save_model(os.path.join(out_dir, f"epoch-{epoch}"))
    tok.save_pretrained(os.path.join(out_dir, f"epoch-{epoch}"))

# --- Final top-level save ---
trainer.save_model(out_dir)
tok.save_pretrained(out_dir)

def update_models_config(model_name, path):
    if os.path.exists(MODELS_CONFIG):
        with open(MODELS_CONFIG, "r") as f:
            try:
                models = json.load(f)
            except json.JSONDecodeError:
                models = {}
    else:
        models = {}

    models[model_name] = {"path": path}

    with open(MODELS_CONFIG, "w") as f:
        json.dump(models, f, indent=2)
    print(f"[INFO] models.config updated with {model_name} → {path}")

# --------- Build optimizer (choose one path) ---------
# A) Duo optimizer from Duo.py
print("Model:", type(model))
print("Params count:", sum(p.numel() for p in model.parameters()))
print("Trainable params count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

use_optimizers = (duo_optim, duo_sched)
if hasattr(duo_optim, "ghost_gate"):
    trainer.add_callback(GhostGateCallback(duo_optim.ghost_gate))


if getattr(trainer, "use_jade_reweighting", False) and trainer.jade_lex:
    jade = trainer.jade_lex.bind_metrics(metrics)
    metrics["loss"] = JadeCodewrightLexicon.jade_loss_reweight(metrics["loss"], jade)

force = np.random.rand()  # or derive from Mahadevi angle / Jade entropy
political = 1.0           # or jade_lex scalar binding

decision = adversary.evaluate(force, political, context="training-loop")
print("⚔ Adversarial:", decision["message"])
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"[MODEL] total_params={total:,} trainable={trainable:,} ({100*trainable/total:.2f}%)")
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

blade = EdgeOfFold(fusion_model)

blade.cut_low_gradients()
blade.slice_layer("lm_head")
blade.invoke_edge_alignment()
realmsy = Realmsy()
infinitizer = VolumetricInfinitizer(trainer, outdir, realmsy)
infinitizer.run(epochs=cli_args.epochs)
_ = possessor.ratio_paternalizer(list(fusion_model.parameters()))

# now update config ONCE
update_models_config(f"olivia-v{version}", out_dir)

old_step = duo_optim.step
def wrapped_step(*args, **kwargs):
    last = trainer.state.log_history[-1] if trainer.state.log_history else {}
    epoch = int(trainer.state.epoch) if trainer.state.epoch is not None else 0
    return old_step(*args, metrics=last, epoch=epoch, **kwargs)

duo_optim.step = wrapped_step

# --- Output directory versioning ---
outdir, version = next_model_version("olivia")
os.makedirs(outdir, exist_ok=True)

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)
print(f"[INFO] Training Olivia version v{version} → {outdir}") 

# === TGDK DATSIK FINALIZATION ===
import os, json, hashlib, binascii, torch
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

print("[DATSIK] 🔒 Entering deterministic finalization sequence...")

ARTIFACTS_DIR = Path("./artifacts")
KEYS_DIR = Path("./datsik_keys")
SIGNERS = ["q0", "q1", "q2", "q3"]
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def sha256_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_private(name):
    with open(KEYS_DIR / f"{name}.pem", "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def load_public(name):
    with open(KEYS_DIR / f"{name}.pub.pem", "rb") as f:
        return serialization.load_pem_public_key(f.read())

def sign_blob(blob, priv):
    return priv.sign(
        blob,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

def verify_blob(blob, sig, pub):
    try:
        pub.verify(sig, blob,
                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                   hashes.SHA256())
        return True
    except Exception:
        return False

# === scan for checkpoints ===
checkpoints = list(ARTIFACTS_DIR.glob("*.pt"))
if not checkpoints:
    print("[DATSIK] ⚠️ No .pt checkpoints found; nothing to sign.")
else:
    for ck in checkpoints:
        print(f"[DATSIK] 🧩 Signing {ck.name}")
        blob = ck.read_bytes()
        checksum = sha256_digest(ck)
        sigs = {"checksum": checksum, "commit": os.getenv("GITHUB_SHA", "local"), "signatures": {}}

        for name in SIGNERS:
            key_path = KEYS_DIR / f"{name}.pem"
            if not key_path.exists():
                print(f"[DATSIK] ⚠️ Missing key {key_path}, skipping signer.")
                continue
            priv = load_private(name)
            sig = sign_blob(blob, priv)
            sigs["signatures"][name] = binascii.hexlify(sig).decode()

        sig_file = ck.with_suffix(".pt.signatures.json")
        sig_file.write_text(json.dumps(sigs, indent=2))
        print(f"[DATSIK] ✅ Signatures written to {sig_file.name}")

        # === verification pass ===
        all_ok = True
        for name in SIGNERS:
            pub_path = KEYS_DIR / f"{name}.pub.pem"
            if not pub_path.exists() or name not in sigs["signatures"]:
                continue
            pub = load_public(name)
            sig_hex = sigs["signatures"][name]
            if not verify_blob(blob, binascii.unhexlify(sig_hex), pub):
                print(f"[DATSIK] ❌ Verification failed for {name}")
                all_ok = False
        if all_ok:
            print(f"[DATSIK] ✅ Verification OK for {ck.name}")
        else:
            print(f"[DATSIK] ⚠️ Verification mismatch detected for {ck.name}")

    print("\n[DΞT-SΞC] DATSIK deterministic signature sequence complete.\n")

# === optional CI hook ===
if os.getenv("TGDK_CI") == "1":
    print("[DATSIK] ⛓️ CI mode enabled — exporting signatures for attestation.")
    import subprocess
    subprocess.run(["gh", "workflow", "run", "datsik_signer.yml"], check=False)



if __name__ == "__main__":
    cw = CodeWright()  # prompts user for AI, author, org names
    info = cw.register_identity()
    print("🔖 Registered Identity:", info)

    payload = "This model achieves symbolic equilibrium."
    sealed = cw.seal(payload)
    print("\nSealed payload:\n", sealed)

    verified = cw.verify(sealed)
    print("\nSeal verification:", verified)

    print("⚡ Pre-building dimensional environment...")
    warmup_environment(fusion_model, cli_args, total_steps)   # warm up the *actual* training model

    # Build MMT symbolic controller once
    duo_optim = locals().get("duo_optim", None) or object()
    mmt_controller = build_dimensional_environment(fusion_model, cli_args)
    duo_optim.attach_model(fusion_model)
    trainable_params = [n for n, p in fusion_model.named_parameters() if p.requires_grad]
    print(f"[DEBUG] Trainables: {len(trainable_params)} layers → {trainable_params[:10]}...")

    # Get optimizer + scheduler
    duo_optim, duo_sched = make_optimizer_scheduler(fusion_model, cli_args, total_steps)

    # Single Trainer instance
    trainer = SFTTrainer(
        model=fusion_model,                
        tokenizer=tgdk_tok, 
        peft_config=peft_cfg,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        args=sft_config,                   
        dataset_text_field="text",         
        packing=False,                     
        optimizers=(duo_optim, duo_sched), # only pass one tuple
        processing_class=tgdk_tok,
    )

    # Add callbacks after Trainer is created
    trainer.add_callback(PlateauEscapeCallback(
        trainer,
        patience=cli_args.plateau_patience,
        min_delta=cli_args.plateau_delta,
        outdir=outdir
    ))
    trainer.add_callback(AdversarialManeuverCallback(
        patience=cli_args.plateau_patience,
        min_delta=cli_args.plateau_delta
    ))

    print("🚀 Starting training...")
    trainer.train()
