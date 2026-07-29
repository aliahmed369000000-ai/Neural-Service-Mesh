"""
ai/preview_check.py
====================
🆕 المرحلة 4 من خطة "Replit Agent Level" — معاينة وتحقّق بصري.

NSM تطبيق Streamlit، فلا يوجد معاينة مرئية مباشرة داخل الوكيل مثل
Replit Agent. البديل العملي المطبَّق هنا: تشغيل `streamlit run` فعلياً
في عملية خلفية مؤقتة على منفذ محلي حر، ثم التأكد أن الصفحة تُحمَّل
فعلياً (لا خطأ 500، والعملية لم تنهَر عند الإقلاع) قبل اعتبار أي مهمة
منجزة فعلاً — بدل الاكتفاء بنجاح `py_compile` (المرحلة 1) الذي يفحص
syntax فقط ولا يكتشف أخطاء وقت التشغيل (استيراد ناقص، استثناء عند
الإقلاع، إلخ).

الاستخدام النموذجي:
    from ai.preview_check import check_streamlit_boots
    result = check_streamlit_boots("streamlit_app.py")
    if result.startswith("✅"):
        ...
"""
from __future__ import annotations

import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

_BOOT_TIMEOUT_SECONDS  = 25   # أقصى وقت انتظار حتى تستجيب الصفحة
_POLL_INTERVAL_SECONDS = 0.7
_MAX_ERROR_SNIPPET     = 1_200


def _free_local_port() -> int:
    """يحجز منفذاً محلياً حراً فعلياً (لا رقماً ثابتاً قد يكون مشغولاً)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def check_streamlit_boots(
    entry_file: str = "streamlit_app.py",
    timeout: int = _BOOT_TIMEOUT_SECONDS,
) -> str:
    """
    يُشغّل `streamlit run <entry_file>` في عملية خلفية مؤقتة على منفذ حر،
    ينتظر حتى تستجيب الصفحة أو تنتهي المهلة، ثم يُنهي العملية دائماً.

    يعيد نصاً يبدأ بـ:
      "✅" — الصفحة حُمِّلت بنجاح (HTTP 200، والعملية لم تنهَر)
      "❌" — فشل حقيقي: خطأ 500، أو انهيار العملية عند الإقلاع، أو
             انتهت المهلة دون أي استجابة (تعليق/حلقة لا نهائية محتملة)

    لا يرمي استثناءً أبداً — أي خطأ غير متوقع يُعاد كنص "❌" ليدخل نفس
    مسار self-healing/منع الرفع الموجود أصلاً في المرحلتين 1 و3.
    """
    entry_path = ROOT / entry_file
    if not entry_path.exists():
        return f"❌ خطأ في المعاينة الحيّة: الملف غير موجود: {entry_file}"

    try:
        port = _free_local_port()
    except Exception as e:
        return f"❌ خطأ في المعاينة الحيّة: تعذّر حجز منفذ محلي: {e}"

    log_path = ROOT / f".preview_check_{port}.log"
    proc: Optional[subprocess.Popen] = None

    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                [
                    "streamlit", "run", str(entry_path),
                    "--server.port", str(port),
                    "--server.address", "127.0.0.1",
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false",
                    "--server.runOnSave", "false",
                ],
                cwd=str(ROOT),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )

            url = f"http://127.0.0.1:{port}/"
            deadline = time.monotonic() + timeout
            last_status: Optional[int] = None
            last_error: Optional[str] = None

            while time.monotonic() < deadline:
                # ── العملية انهارت قبل أي استجابة ناجحة = فشل حقيقي ──
                if proc.poll() is not None:
                    break

                try:
                    with urllib.request.urlopen(url, timeout=3) as resp:
                        last_status = resp.status
                        body = resp.read(4000).decode("utf-8", errors="replace")
                        if last_status and last_status < 500:
                            note = ""
                            if "Traceback (most recent call last)" in body:
                                note = (
                                    " ⚠️ ملاحظة: الصفحة حُمِّلت (200) لكن يبدو أن "
                                    "هناك استثناء Python ظاهراً داخل واجهة "
                                    "Streamlit نفسها — راجع محتوى الصفحة يدوياً."
                                )
                            return (
                                f"✅ المعاينة الحيّة: `{entry_file}` يُحمَّل بنجاح "
                                f"(HTTP {last_status}) على المنفذ {port}.{note}"
                            )
                        else:
                            last_error = body[:_MAX_ERROR_SNIPPET]
                            break
                except urllib.error.HTTPError as e:
                    last_status = e.code
                    if e.code >= 500:
                        try:
                            last_error = e.read().decode("utf-8", errors="replace")[:_MAX_ERROR_SNIPPET]
                        except Exception:
                            last_error = str(e)
                        break
                    # حالة نادرة: كود 4xx قبل اكتمال إقلاع Streamlit — أعد المحاولة
                except (urllib.error.URLError, ConnectionError, OSError):
                    pass  # الخادم لم يقلع بعد — طبيعي في الثواني الأولى

                time.sleep(_POLL_INTERVAL_SECONDS)

            # ── لم نصل لعودة ناجحة داخل الحلقة أعلاه ──
            if last_status is not None and last_status >= 500:
                return (
                    f"❌ خطأ في المعاينة الحيّة: `{entry_file}` يرجع HTTP "
                    f"{last_status} (خطأ خادم) عند التحميل:\n```\n{last_error}\n```"
                )

            if proc.poll() is not None:
                # العملية خرجت قبل أن تستجيب أبداً — نقرأ آخر سطور اللوق
                try:
                    log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-_MAX_ERROR_SNIPPET:]
                except Exception:
                    log_tail = "(تعذّرت قراءة سجل التشغيل)"
                return (
                    f"❌ خطأ في المعاينة الحيّة: عملية `streamlit run {entry_file}` "
                    f"انهارت عند الإقلاع (رمز الخروج {proc.returncode}):\n"
                    f"```\n{log_tail}\n```"
                )

            return (
                f"❌ خطأ في المعاينة الحيّة: انتهت المهلة ({timeout}ث) دون أي "
                f"استجابة HTTP من `{entry_file}` على المنفذ {port} — قد تكون "
                f"هناك حلقة لا نهائية أو تعليق عند الإقلاع."
            )

    except FileNotFoundError:
        return ("❌ خطأ في المعاينة الحيّة: أمر `streamlit` غير متاح في هذه "
                "البيئة (تأكد من تثبيت الحزمة).")
    except Exception as e:
        return f"❌ خطأ في المعاينة الحيّة: {e}"
    finally:
        # ── إنهاء العملية دائماً، مهما كانت النتيجة ──
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass
