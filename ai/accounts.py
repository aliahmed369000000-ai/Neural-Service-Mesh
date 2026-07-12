"""
accounts.py — نظام حسابات عام بسيط (اسم مستخدم + كلمة مرور)
==============================================================
يُستخدم أولاً من واجهة Streamlit (تسجيل/دخول)، ومصمَّم من البداية بحيث
يقبل ربط رقم هاتف (phone_number) بالحساب لاحقاً — هذا هو المعرّف الذي
ستستخدمه بوابة واتساب (webhook منفصل) لمطابقة رسالة واردة بحساب موجود،
بدل بناء نظام حسابات ثانٍ مخصص لواتساب.

لا اعتماديات خارجية: تشفير كلمة المرور عبر hashlib.pbkdf2_hmac (مكتبة
قياسية في بايثون) بدل bcrypt/passlib — تفادياً لأي تثبيت pip إضافي.

قاعدة البيانات: memory/accounts.db (ملف SQLite جديد ومستقل، لا يلمس أي
قاعدة بيانات موجودة بالمشروع).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "memory" / "accounts.db"
DB_PATH.parent.mkdir(exist_ok=True)

_PBKDF2_ITERATIONS = 260_000
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\.\u0600-\u06FF]{3,32}$")  # يدعم عربي/إنجليزي


class AccountError(ValueError):
    """خطأ متوقع بمنطق الحسابات (اسم مستخدم مكرر، كلمة مرور ضعيفة، ...) —
    يُعرض نصّه مباشرة للمستخدم، عكس استثناءات الأخطاء البرمجية غير المتوقعة."""


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            phone_number TEXT UNIQUE,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        )""")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone_number)"
    )
    conn.commit()
    return conn


def _hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """يعيد (hash_hex, salt_hex). لو salt غير معطى يُولَّد عشوائياً (تسجيل جديد)."""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return digest.hex(), salt.hex()


def create_user(username: str, password: str, phone_number: Optional[str] = None) -> int:
    """ينشئ حساباً جديداً. يرفع AccountError برسالة عربية واضحة لو:
    اسم المستخدم غير صالح/مكرر، كلمة المرور قصيرة، أو الهاتف مستخدم مسبقاً."""
    username = username.strip()
    if not _USERNAME_RE.match(username):
        raise AccountError(
            "اسم المستخدم يجب أن يكون 3-32 حرفاً (أحرف/أرقام/عربي فقط)"
        )
    if len(password) < 8:
        raise AccountError("كلمة المرور يجب أن تكون 8 أحرف على الأقل")

    phone_number = phone_number.strip() if phone_number else None

    password_hash, salt = _hash_password(password)
    now = datetime.now(timezone.utc).isoformat()

    with _db() as c:
        try:
            cur = c.execute(
                "INSERT INTO users (username, password_hash, salt, phone_number, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, salt, phone_number, now),
            )
            c.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError as exc:
            if "username" in str(exc):
                raise AccountError("اسم المستخدم هذا مستخدم بالفعل") from exc
            if "phone_number" in str(exc):
                raise AccountError("رقم الهاتف هذا مرتبط بحساب آخر بالفعل") from exc
            raise AccountError("تعذّر إنشاء الحساب") from exc


def verify_login(username: str, password: str) -> Optional[Dict]:
    """يتحقق من اسم المستخدم/كلمة المرور. يعيد بيانات المستخدم (بدون
    password_hash/salt) عند النجاح، أو None عند الفشل — لا يرفع استثناء
    أبداً (فشل تسجيل الدخول حالة متوقعة، ليست خطأ برمجياً)."""
    with _db() as c:
        row = c.execute(
            "SELECT id, username, password_hash, salt, phone_number, created_at "
            "FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if not row:
        return None
    uid, uname, stored_hash, salt_hex, phone, created_at = row
    computed_hash, _ = _hash_password(password, bytes.fromhex(salt_hex))
    if not hmac.compare_digest(computed_hash, stored_hash):
        return None

    now = datetime.now(timezone.utc).isoformat()
    with _db() as c:
        c.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, uid))
        c.commit()

    return {"id": uid, "username": uname, "phone_number": phone, "created_at": created_at}


def get_user_by_phone(phone_number: str) -> Optional[Dict]:
    """يُستخدم من بوابة واتساب لاحقاً لمطابقة رقم هاتف وارد بحساب موجود."""
    with _db() as c:
        row = c.execute(
            "SELECT id, username, phone_number, created_at, last_login_at "
            "FROM users WHERE phone_number = ?",
            (phone_number.strip(),),
        ).fetchone()
    if not row:
        return None
    uid, uname, phone, created_at, last_login = row
    return {
        "id": uid, "username": uname, "phone_number": phone,
        "created_at": created_at, "last_login_at": last_login,
    }


def link_phone(user_id: int, phone_number: str) -> None:
    """يربط رقم هاتف بحساب موجود (يُستخدم لاحقاً من تدفّق ربط واتساب).
    يرفع AccountError لو الرقم مرتبط بحساب آخر مسبقاً."""
    phone_number = phone_number.strip()
    with _db() as c:
        try:
            c.execute(
                "UPDATE users SET phone_number = ? WHERE id = ?",
                (phone_number, user_id),
            )
            c.commit()
        except sqlite3.IntegrityError as exc:
            raise AccountError("رقم الهاتف هذا مرتبط بحساب آخر بالفعل") from exc
