"""
core_config.py — TGDK Core Configuration Loader
------------------------------------------------
Provides argument parsing, environment handling, and model configuration
for OliviaAI / TGDK LLM fine-tuning pipelines.
"""

import os
import json
import argparse
import logging


def parse_cli_args():
    """
    🔧 Parse command-line arguments for TGDK QLoRA training.
    Returns a namespace object (args).
    """
    parser = argparse.ArgumentParser(description="TGDK / OliviaAI QLoRA Runner")

    parser.add_argument("--config", type=str, default="config.tgdkcfg",
                        help="Path to model config file (.json or .tgdkcfg)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max token sequence length")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda / cpu)")
    parser.add_argument("--out_dir", type=str, default="./out", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoint")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # --- Logging setup ---
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[core_config] %(message)s"
    )

    logging.info(f"[core_config] Loaded CLI args → epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    return args


def load_model_config(path: str = None) -> dict:
    """
    🧩 Load TGDK model configuration file.
    Supports .json or custom .tgdkcfg format.
    Returns a dict with training + model hyperparameters.
    """
    # === Default path ===
    if path is None:
        path = os.environ.get("TGDKCFG_PATH", "config.tgdkcfg")

    cfg = {
        "epochs": 1,
        "batch_size": 1,
        "lr": 2e-5,
        "max_seq_length": 2048,
        "model_name": "mistralai/Mistral-7B-v0.1",
        "output_dir": "./out",
        "use_lora": True,
        "save_steps": 500,
        "logging_steps": 50,
    }

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Try JSON first
                if content.startswith("{"):
                    file_cfg = json.loads(content)
                else:
                    # Simple key=value fallback for .tgdkcfg
                    file_cfg = {}
                    for line in content.splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            file_cfg[k.strip()] = v.strip()
                cfg.update(file_cfg)
                logging.info(f"[core_config] Loaded config from {path}")
        except Exception as e:
            logging.warning(f"[core_config] Failed to load {path}: {e}")
    else:
        logging.warning(f"[core_config] Config file not found at {path}, using defaults.")

    return cfg


if __name__ == "__main__":
    # Quick standalone test
    args = parse_cli_args()
    cfg = load_model_config(args.config)
    print(json.dumps(cfg, indent=2))
