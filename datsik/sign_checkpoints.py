"""
sign_checkpoints.py
-------------------
CI Quad-Signer for DATSIK model checkpoints.
- Signs all .pt files in ./artifacts
- Produces .signatures.json per checkpoint
- Verifies signatures against public keys
"""

import os, json, hashlib, binascii
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
KEYS_DIR = Path(os.getenv("TGDK_KEYS_DIR", "./datsik_keys"))
print(f"[CI-Signer] Using keys from {KEYS_DIR}")

def load_private_key(name: str):
    with open(KEYS_DIR / f"{name}.pem", "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def load_public_key(name: str):
    with open(KEYS_DIR / f"{name}.pub.pem", "rb") as f:
        return serialization.load_pem_public_key(f.read())

def sign_blob(blob: bytes, priv) -> bytes:
    return priv.sign(
        blob,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

def verify_blob(blob: bytes, sig: bytes, pub) -> bool:
    try:
        pub.verify(
            sig,
            blob,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False

def sha256_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

signers = ["q0", "q1", "q2", "q3"]

for ck in ARTIFACTS_DIR.glob("*.pt"):
    print(f"\n[CI-Signer] Signing {ck.name}")
    blob = ck.read_bytes()
    checksum = sha256_digest(ck)
    signatures = {"checksum": checksum, "commit": os.getenv("GITHUB_SHA", "local"), "signatures": {}}

    for name in signers:
        priv = load_private_key(name)
        sig = sign_blob(blob, priv)
        signatures["signatures"][name] = binascii.hexlify(sig).decode()

    sig_file = ck.with_suffix(".pt.signatures.json")
    sig_file.write_text(json.dumps(signatures, indent=2))
    print(f"✅ Wrote {sig_file.name}")

    # verification step
    verified = True
    for name in signers:
        pub = load_public_key(name)
        sig_hex = signatures["signatures"][name]
        if not verify_blob(blob, binascii.unhexlify(sig_hex), pub):
            print(f"❌ Verification failed for {name}")
            verified = False
    print("✅ Verification OK" if verified else "❌ Verification mismatch")

print("\n[CI-Signer] All checkpoints processed.")
