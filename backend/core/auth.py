"""
Auth helpers — password hashing + JWT tokens.
Uses only Python stdlib + python-jose + passlib (already common deps).
"""
import os
import hashlib
import hmac
import base64
import json
import time

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production-please")

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return base64.b64encode(salt + key).decode()

def verify_password(password: str, stored: str) -> bool:
    try:
        decoded = base64.b64decode(stored.encode())
        salt, key = decoded[:16], decoded[16:]
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return hmac.compare_digest(key, new_key)
    except Exception:
        return False

def create_token(user_id: int, username: str) -> str:
    payload = {"user_id": user_id, "username": username, "exp": int(time.time()) + 86400 * 7}
    data = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
    return f"{data}.{sig}"

def decode_token(token: str):
    try:
        data, sig = token.rsplit(".", 1)
        expected = hmac.new(SECRET_KEY.encode(), data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(data.encode() + b"=="))
        if payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None
