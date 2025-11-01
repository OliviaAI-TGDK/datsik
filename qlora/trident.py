from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import os

os.makedirs("/mnt/data/datsik_keys", exist_ok=True)

key_paths = []
for i in range(4):
    name = f"q{i}"
    priv_path = f"/mnt/data/datsik_keys/{name}.pem"
    pub_path = f"/mnt/data/datsik_keys/{name}.pub.pem"
    # generate 4096-bit RSA key
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    priv_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,  # PKCS#1
        encryption_algorithm=serialization.NoEncryption()
    )
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open(priv_path, "wb") as f:
        f.write(priv_pem)
    with open(pub_path, "wb") as f:
        f.write(pub_pem)
    key_paths.append((priv_path, pub_path))

# List created files
for priv, pub in key_paths:
    print(priv)
    print(pub)
