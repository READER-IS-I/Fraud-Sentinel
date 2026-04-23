from __future__ import annotations

from base64 import b64decode, b64encode
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.utils.file_utils import ensure_dir, load_json, save_json


def file_sha256(path: Path | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_rsa_keypair(private_key_path: Path | str, public_key_path: Path | str) -> tuple[Path, Path]:
    private_path = Path(private_key_path)
    public_path = Path(public_key_path)
    ensure_dir(private_path.parent)
    ensure_dir(public_path.parent)
    if private_path.exists() and public_path.exists():
        return private_path, public_path
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def encrypt_file(input_path: Path | str, output_dir: Path | str, public_key_path: Path | str) -> dict:
    source = Path(input_path)
    target_dir = ensure_dir(output_dir)
    public_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    raw_data = source.read_bytes()
    digest = sha256(raw_data).hexdigest()
    symmetric_key = Fernet.generate_key()
    encrypted_data = Fernet(symmetric_key).encrypt(raw_data)
    wrapped_key = public_key.encrypt(
        symmetric_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    encrypted_path = target_dir / f"{source.stem}.fsenc"
    meta_path = target_dir / f"{source.stem}.meta.json"
    encrypted_path.write_bytes(encrypted_data)
    save_json(
        meta_path,
        {
            "original_name": source.name,
            "algorithm": "Fernet + RSA-OAEP",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sha256": digest,
            "wrapped_key": b64encode(wrapped_key).decode("utf-8"),
        },
    )
    return {
        "original_path": str(source),
        "encrypted_path": str(encrypted_path),
        "meta_path": str(meta_path),
        "sha256": digest,
    }


def decrypt_file(encrypted_path: Path | str, meta_path: Path | str, private_key_path: Path | str, output_path: Path | str | None = None) -> dict:
    encrypted_file = Path(encrypted_path)
    metadata = load_json(meta_path)
    private_key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    symmetric_key = private_key.decrypt(
        b64decode(metadata["wrapped_key"]),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    decrypted_bytes = Fernet(symmetric_key).decrypt(encrypted_file.read_bytes())
    digest = sha256(decrypted_bytes).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("文件完整性校验失败，SHA-256 摘要不匹配。")
    target = Path(output_path) if output_path else encrypted_file.with_name(f"{encrypted_file.stem}_decrypted.pt")
    ensure_dir(target.parent)
    target.write_bytes(decrypted_bytes)
    return {
        "encrypted_path": str(encrypted_file),
        "decrypted_path": str(target),
        "sha256": digest,
        "status": "success",
    }
