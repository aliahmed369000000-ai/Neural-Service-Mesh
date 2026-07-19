"""
accounts.py — نظام حسابات عام بسيط (اسم مستخدم + كلمة مرور)
==============================================================
يُستخدم أولاً من واجهة Streamlit (تسجيل/دخول)، ومصمَّم من البداية بحيث
يقبل ربط رقم هاتف (phone_number) بالحساب لاحقاً — هذا هو المعرّف الذي
ستستخدمه بوابة واتساب (webhook منفصل) لمطابقة رسالة واردة بحساب موجود،
بدل بناء نظام حسابات ثانٍ مخصص لواتساب.

لا اعتماديات خارجية إضافية عن المشروع: تشفير كلمة المرور عبر
hashlib.pbkdf2_hmac (مكتبة قياسية)، وrequests (موجودة أصلاً بـ
requirements.txt) لمزامنة Upstash الاختيارية أدناه.

قاعدة البيانات الأساسية: memory/accounts.db (SQLite، مصدر الحقيقة
الوحيد لواجهة Streamlit — لا شي هنا يغيّر ذلك).

⚠️ ملاحظة معمارية مهمة (اكتُشفت أثناء تصميم بوابة واتساب):
memory/accounts.db يعيش على قرص Streamlit Community Cloud فقط. أي
خدمة خارجية (مثل دالة Vercel لبوابة واتساب) لا تقدر تصل له إطلاقاً —
نظامي ملفات منفصلين تماماً. لذلك عند توفّر رقم هاتف، نزامن نسخة خفيفة
(اسم المستخدم + تاريخ الإنشاء فقط، بدون كلمة المرور) لـUpstash Redis
بالتوازي — هذا هو المصدر اللي تقرأ منه بوابة واتساب. المزامنة اختيارية
تماماً وصامتة الفشل (best-effort): لو متغيرات البيئة غير مضبوطة أو
فشل الاتصال، إنشاء/تعديل الحساب بـStreamlit ينجح بشكل طبيعي كالمعتاد
دون أي تأثير — فقط ميزة "حالة حسابي" بواتساب لن تعرف عن الحساب حتى
تنجح المزامنة لاحقاً.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "memory" / "accounts.db"
DB_PATH.parent.mkdir(exist_ok=True)

_PBKDF2_ITERATIONS = 260_000
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\.\u0600-\u06FF]{3,32}$")  # يدعم عربي/إنجليزي
_UPSTASH_HTTP_TIMEOUT = 5


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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            username      TEXT PRIMARY KEY,
            failed_count  INTEGER NOT NULL DEFAULT 0,
            locked_until  TEXT
        )""")
    conn.commit()
    return conn


# ── حماية من محاولات الدخول المتكررة (brute-force) ─────────────────────────
MAX_FAILED_ATTEMPTS = 5      # عدد المحاولات الفاشلة المسموحة قبل القفل
LOCKOUT_MINUTES = 15         # مدة القفل المؤقت بالدقائق


def _check_lockout(c: sqlite3.Connection, username: str) -> None:
    """يرفع AccountError إن كان الحساب مقفلاً حالياً. لا يفعل شيئاً غير ذلك."""
    row = c.execute(
        "SELECT locked_until FROM login_attempts WHERE username = ?", (username,)
    ).fetchone()
    if not row or not row[0]:
        return
    locked_until = datetime.fromisoformat(row[0])
    now = datetime.now(timezone.utc)
    if now < locked_until:
        remaining_min = max(1, int((locked_until - now).total_seconds() // 60) + 1)
        raise AccountError(
            f"⛔ تم قفل الحساب مؤقتاً بسبب محاولات دخول فاشلة متكررة. "
            f"حاول مرة أخرى بعد {remaining_min} دقيقة."
        )


def _record_failed_attempt(c: sqlite3.Connection, username: str) -> None:
    """يزيد عدّاد الفشل، ويقفل الحساب مؤقتاً عند بلوغ الحد الأقصى."""
    row = c.execute(
        "SELECT failed_count FROM login_attempts WHERE username = ?", (username,)
    ).fetchone()
    failed_count = (row[0] if row else 0) + 1

    if failed_count >= MAX_FAILED_ATTEMPTS:
        locked_until = (
            datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
        ).isoformat()
        c.execute(
            """INSERT INTO login_attempts (username, failed_count, locked_until)
               VALUES (?, ?, ?)
               ON CONFLICT(username) DO UPDATE SET failed_count = ?, locked_until = ?""",
            (username, 0, locked_until, 0, locked_until),
        )
    else:
        c.execute(
            """INSERT INTO login_attempts (username, failed_count, locked_until)
               VALUES (?, ?, NULL)
               ON CONFLICT(username) DO UPDATE SET failed_count = ?""",
            (username, failed_count, failed_count),
        )
    c.commit()


def _clear_failed_attempts(c: sqlite3.Connection, username: str) -> None:
    c.execute(
        """INSERT INTO login_attempts (username, failed_count, locked_until)
           VALUES (?, 0, NULL)
           ON CONFLICT(username) DO UPDATE SET failed_count = 0, locked_until = NULL""",
        (username,),
    )
    c.commit()


def _hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
    """يعيد (hash_hex, salt_hex). لو salt غير معطى يُولَّد عشوائياً (تسجيل جديد)."""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return digest.hex(), salt.hex()


def _sync_phone_to_upstash(phone_number: str, username: str, created_at: str) -> None:
    """مزامنة اختيارية صامتة الفشل — تكتب {username, created_at} إلى
    Upstash Redis تحت مفتاح مبني من رقم الهاتف، عشان بوابة واتساب
    (دالة Vercel منفصلة، بدون وصول لـmemory/accounts.db) تقدر تعرف
    إن هذا الرقم مرتبط بحساب. لا ترفع أي استثناء أبداً — فشلها لا يجب
    أن يمنع إنشاء/تعديل الحساب الأساسي بـStreamlit."""
    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if not url or not token:
        return
    try:
        import requests

        key = "wa_account_phone:" + quote(phone_number.strip(), safe="")
        value = quote(json.dumps({"username": username, "created_at": created_at}), safe="")
        requests.post(
            f"{url.rstrip('/')}/set/{key}/{value}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_UPSTASH_HTTP_TIMEOUT,
        )
    except Exception:
        pass  # صامت عمداً — راجع الملاحظة المعمارية أعلى الملف


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
            if phone_number:
                _sync_phone_to_upstash(phone_number, username, now)
            return cur.lastrowid
        except sqlite3.IntegrityError as exc:
            if "username" in str(exc):
                raise AccountError("اسم المستخدم هذا مستخدم بالفعل") from exc
            if "phone_number" in str(exc):
                raise AccountError("رقم الهاتف هذا مرتبط بحساب آخر بالفعل") from exc
            raise AccountError("تعذّر إنشاء الحساب") from exc


def verify_login(username: str, password: str) -> Optional[Dict]:
    """يتحقق من اسم المستخدم/كلمة المرور. يعيد بيانات المستخدم (بدون
    password_hash/salt) عند النجاح، أو None عند فشل بيانات الدخول
    (حالة متوقعة). يرفع AccountError فقط عند قفل الحساب مؤقتاً بسبب
    محاولات فاشلة متكررة (حماية brute-force)."""
    username = username.strip()
    with _db() as c:
        _check_lockout(c, username)  # يرفع AccountError إن كان مقفلاً

        row = c.execute(
            "SELECT id, username, password_hash, salt, phone_number, created_at "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if not row:
            _record_failed_attempt(c, username)
            return None

        uid, uname, stored_hash, salt_hex, phone, created_at = row
        computed_hash, _ = _hash_password(password, bytes.fromhex(salt_hex))
        if not hmac.compare_digest(computed_hash, stored_hash):
            _record_failed_attempt(c, username)
            return None

        _clear_failed_attempts(c, username)
        now = datetime.now(timezone.utc).isoformat()
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
            row = c.execute(
                "SELECT username, created_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if row:
                _sync_phone_to_upstash(phone_number, row[0], row[1])
        except sqlite3.IntegrityError as exc:
            raise AccountError("رقم الهاتف هذا مرتبط بحساب آخر بالفعل") from exc
