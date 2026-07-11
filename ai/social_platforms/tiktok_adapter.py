"""
محول TikTok — عبر Content Posting API الرسمي (Direct Post / FILE_UPLOAD).
https://developers.tiktok.com/doc/content-posting-api-get-started/

⚠️ قيد مهم وحقيقي (ليس افتراضاً): تطبيقات TikTok غير المُدقَّقة (unaudited —
أي أغلب تطبيقات المطورين الأفراد قبل مراجعة TikTok الرسمية لها) يمكنها فقط
النشر بخصوصية "SELF_ONLY" (يظهر الفيديو كمسودة/خاص بحساب صاحب التوكن فقط،
وليس عاماً). النشر العام (PUBLIC_TO_EVERYONE) يتطلب أن يجتاز التطبيق مراجعة
TikTok Content Posting API Audit. هذا القيد من TikTok نفسه وليس نقصاً بهذا
الكود — هذا المحول يستدعي /content/check/ أولاً لمعرفة القيود المتاحة فعلياً
لتوكن الحساب قبل النشر.

يتطلب: TIKTOK_CLIENT_KEY، TIKTOK_CLIENT_SECRET، TIKTOK_OAUTH_REFRESH_TOKEN
(يُولَّد refresh token مرة عبر تدفق OAuth الكامل خارج هذا الملف — تسجيل
دخول المستخدم عبر متصفح لمرة واحدة ثم تخزين الـrefresh token في البيئة).
النطاق (scope) المطلوب: video.publish
"""

from __future__ import annotations

import os
from typing import List, Optional

import requests

from .base import PlatformAdapter, SocialItem, NotConfiguredError

TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
CREATOR_INFO_URL = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"


class TikTokAdapter(PlatformAdapter):
    platform_id = "tiktok"
    required_env = ["TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "TIKTOK_OAUTH_REFRESH_TOKEN"]

    # TikTok لا يوفّر API عام لجلب التعليقات/المنشنات لحساب فردي (بخلاف
    # يوتيوب/تويتر) — فقط Content Posting API للنشر. لذا fetch_new_items
    # و reply غير مدعومتين هنا عمداً (نرفع NotImplementedError بدل تلفيق شيء).

    def _access_token(self) -> str:
        self._require_configured()
        r = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Cache-Control": "no-cache"},
            data={
                "client_key": os.environ["TIKTOK_CLIENT_KEY"],
                "client_secret": os.environ["TIKTOK_CLIENT_SECRET"],
                "grant_type": "refresh_token",
                "refresh_token": os.environ["TIKTOK_OAUTH_REFRESH_TOKEN"],
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if "access_token" not in data:
            raise NotConfiguredError(f"tiktok: فشل تجديد التوكن — {data}")
        return data["access_token"]

    def creator_info(self) -> dict:
        """يجلب قيود النشر الفعلية للحساب (هل خاص/عام، أقصى مدة فيديو،
        هل التعليقات/الدويت/الستيتش مسموحة) — يُستحسن استدعاؤها قبل publish
        لعرض الخيارات الحقيقية المتاحة بدل افتراضها."""
        token = self._access_token()
        r = requests.post(
            CREATOR_INFO_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("data", {})

    def publish(self, text: str) -> str:
        raise NotImplementedError(
            "tiktok: لا يوجد نشر نصي مستقل — استخدم upload_video() لنشر فيديو "
            "(TikTok منصة فيديو فقط عبر Content Posting API)."
        )

    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        raise NotImplementedError(
            "tiktok: لا يوفّر TikTok API عاماً لجلب التعليقات/المنشنات لحساب فردي."
        )

    def reply(self, item: SocialItem, text: str) -> str:
        raise NotImplementedError("tiktok: لا يدعم الرد عبر API عام حالياً.")

    # ── رفع فيديو فعلي (Direct Post — FILE_UPLOAD) ────────────────────────
    def upload_video(
        self,
        video_bytes: bytes,
        title: str,
        privacy_level: Optional[str] = None,
        disable_comment: bool = False,
        disable_duet: bool = False,
        disable_stitch: bool = False,
    ) -> str:
        """يرفع فيديو حقيقي عبر Content Posting API (Direct Post).
        يعيد publish_id (يُستخدم لاحقاً مع check_status() لمعرفة نتيجة
        المعالجة النهائية على TikTok). إن لم يُحدَّد privacy_level، يُستعلم
        تلقائياً عن القيم المتاحة فعلياً للحساب عبر creator_info() ويُختار
        أول خيار متاح (غالباً SELF_ONLY للتطبيقات غير المدقَّقة)."""
        if not video_bytes:
            raise ValueError("tiktok.upload_video: video_bytes فارغة.")

        token = self._access_token()

        if privacy_level is None:
            info = self.creator_info()
            options = info.get("privacy_level_options", ["SELF_ONLY"])
            privacy_level = options[0] if options else "SELF_ONLY"

        video_size = len(video_bytes)
        # حد أقصى شائع لحجم القطعة الواحدة 64MB — نرفع بقطعة واحدة إن كان
        # الفيديو أصغر من ذلك (حالة Shorts النموذجية)، وإلا نقسّمها.
        chunk_size = min(video_size, 64 * 1024 * 1024)
        total_chunks = max(1, -(-video_size // chunk_size))  # ceil division

        init_payload = {
            "post_info": {
                "title": (title or "")[:150],
                "privacy_level": privacy_level,
                "disable_comment": disable_comment,
                "disable_duet": disable_duet,
                "disable_stitch": disable_stitch,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": chunk_size,
                "total_chunk_count": total_chunks,
            },
        }
        init = requests.post(
            INIT_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
            json=init_payload,
            timeout=30,
        )
        init.raise_for_status()
        init_data = init.json()
        if init_data.get("error", {}).get("code") not in (None, "ok"):
            raise RuntimeError(f"tiktok.upload_video: فشلت التهيئة — {init_data['error']}")

        publish_id = init_data["data"]["publish_id"]
        upload_url = init_data["data"]["upload_url"]

        # رفع القطع (chunk واحد غالباً لفيديوهات Shorts القصيرة)
        offset = 0
        for chunk_index in range(total_chunks):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, video_size) - 1
            chunk = video_bytes[start:end + 1]
            put = requests.put(
                upload_url,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Range": f"bytes {start}-{end}/{video_size}",
                },
                data=chunk,
                timeout=900,
            )
            put.raise_for_status()

        return publish_id

    def check_status(self, publish_id: str) -> dict:
        """يستعلم عن حالة معالجة الفيديو بعد الرفع (PROCESSING_UPLOAD →
        PUBLISH_COMPLETE أو FAILED). استدعِها بعد upload_video بثوانٍ."""
        token = self._access_token()
        r = requests.post(
            STATUS_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
            json={"publish_id": publish_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("data", {})
