"""
ai/shared_experience.py
=======================
محرك الخبرة الجماعية (Shared Experience Engine).
يسمح للوكلاء بمشاركة الحقائق عالية الأهمية في قاعدة بيانات موحدة.
"""

import json
import logging
import time
import threading
import base64
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger("NSM.SharedExperience")

class SharedExperienceManager:
    def __init__(self, storage_path: str = "artifacts/learning/shared_knowledge.json", remote_url: Optional[str] = None, encryption_key: Optional[str] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.remote_url = remote_url
        self.api_key = "nsm_secret_key_2026"
        self.knowledge = self._load_knowledge()
        
        # إعداد التشفير
        self.encryption_enabled = encryption_key is not None
        if self.encryption_enabled:
            self.cipher = self._init_cipher(encryption_key)
        else:
            self.cipher = None

    def _init_cipher(self, key_str: str) -> Fernet:
        """إنشاء محرك التشفير من سلسلة نصية."""
        salt = b'nsm_salt_2026'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_str.encode()))
        return Fernet(key)

    def _encrypt(self, text: str) -> str:
        if not self.encryption_enabled or not self.cipher:
            return text
        return self.cipher.encrypt(text.encode()).decode()

    def _decrypt(self, encrypted_text: str) -> str:
        if not self.encryption_enabled or not self.cipher:
            return encrypted_text
        try:
            return self.cipher.decrypt(encrypted_text.encode()).decode()
        except:
            return "[DECRYPTION_FAILED]"

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None):
        """إرسال طلب إلى خادم الذاكرة الموزع."""
        if not self.remote_url:
            return None
        import requests
        url = f"{self.remote_url}{endpoint}"
        headers = {"X-NSM-Token": self.api_key}
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=5)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=5)
            
            if response.status_code != 200:
                logger.error(f"❌ خادم الذاكرة أعاد خطأ {response.status_code}: {response.text}")
                return None
            return response.json()
        except Exception as e:
            logger.error(f"❌ خطأ في الاتصال بخادم الذاكرة: {e}")
            return None
        
    def _load_knowledge(self) -> Dict[str, Any]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"فشل تحميل المعرفة الجماعية: {e}")
        return {"shared_facts": {}, "active_queries": {}, "global_metrics": {}, "version": "1.1"}

    def _save_knowledge(self):
        # إزالة القفل من هنا لتجنب Deadlock إذا تم استدعاؤه من داخل قفل آخر
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"فشل حفظ المعرفة الجماعية: {e}")

    def share_fact(self, agent_id: str, fact: Dict[str, Any]):
        """مشاركة حقيقة مع دعم الوضع الموزع."""
        importance = fact.get("strength", 0)
        content = fact.get("content", "")
        
        if importance < 0.7: return False
            
        # تشفير المحتوى قبل المشاركة
        final_content = self._encrypt(content)

        if self.remote_url:
            res = self._request("POST", "/share", {
                "agent_id": agent_id,
                "content": final_content,
                "importance": importance,
                "semantic_hash": fact.get("semantic_hash"),
                "is_encrypted": self.encryption_enabled
            })
            return res and res.get("status") == "success"

        with self._lock:
            fact_id = f"shared_{uuid.uuid4().hex[:6]}"
            if fact_id not in self.knowledge["shared_facts"]:
                self.knowledge["shared_facts"][fact_id] = {
                    "content": final_content,
                    "origin_agent": agent_id,
                    "shared_at": time.time(),
                    "importance": importance,
                    "verification_count": 1,
                    "semantic_hash": fact.get("semantic_hash"),
                    "is_encrypted": self.encryption_enabled
                }
                self._save_knowledge()
                return True
            return False

    def sync_agent_memory(self, agent_memory: Any):
        """مزامنة ذاكرة الوكيل مع المعرفة الجماعية."""
        new_facts_count = 0
        
        if self.remote_url:
            remote_facts = self._request("GET", f"/sync?agent_id={agent_memory.agent_id}")
            if remote_facts:
                self.knowledge["shared_facts"].update(remote_facts)

        with self._lock:
            knowledge_copy = list(self.knowledge["shared_facts"].items())
            
        for fact_id, fact in knowledge_copy:
            # معالجة حالة أن fact قد يكون قاموساً أو قيمة مباشرة (للتوافق)
            content = fact.get("content") if isinstance(fact, dict) else str(fact)
            if not content: continue
            
            # فك التشفير إذا لزم الأمر
            if isinstance(fact, dict) and fact.get("is_encrypted"):
                content = self._decrypt(content)

            exists = any(f.get("content") == content for f in agent_memory.ltm_semantic.values())
            if not exists:
                importance = fact.get("importance", 0.8) if isinstance(fact, dict) else 0.8
                s_hash = fact.get("semantic_hash") if isinstance(fact, dict) else None
                
                agent_memory.add_fact(
                    content, 
                    semantic_hash=s_hash,
                    importance=importance * 0.9
                )
                new_facts_count += 1
        
        if new_facts_count > 0:
            logger.info(f"📥 تم مزامنة {new_facts_count} حقيقة جماعية للوكيل {agent_memory.agent_id}")
        return new_facts_count

    def ask_swarm(self, agent_id: str, query: str, context: str = ""):
        """طرح سؤال توضيحي على السرب مع توجيه ذكي للخبير."""
        from ai.learning_engine import learning_engine
        domain = learning_engine.classify_domain(query + " " + context)
        
        with self._lock:
            # البحث عن أفضل خبير في هذا المجال
            best_expert = None
            best_score = 0.6 # الحد الأدنى لاعتبار الوكيل خبيراً
            
            for a_id, expertise in learning_engine.trust_scores.items():
                if a_id != agent_id:
                    score = expertise.get(domain, 0)
                    if score > best_score:
                        best_score = score
                        best_expert = a_id
            
            # استخدام UUID لتجنب التصادم عند التزامن العالي
            import uuid
            query_id = f"q_{uuid.uuid4().hex[:6]}"
            self.knowledge["active_queries"][query_id] = {
                "query": query,
                "context": context,
                "asker": agent_id,
                "target_expert": best_expert,
                "domain": domain,
                "timestamp": time.time(),
                "status": "open",
                "answers": []
            }
            # حفظ يدوي داخل القفل لضمان الذرية
            try:
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"فشل حفظ المعرفة الجماعية: {e}")
        
        target_msg = f"موجه إلى {best_expert}" if best_expert else "موجه للجميع"
        logger.info(f"❓ سؤال جديد [{domain}] من {agent_id} ({target_msg}): {query}")
        return query_id

    def answer_query(self, agent_id: str, query_id: str, answer: str):
        """تقديم إجابة لسؤال موجود."""
        if query_id in self.knowledge["active_queries"]:
            self.knowledge["active_queries"][query_id]["answers"].append({
                "answer": answer,
                "provider": agent_id,
                "timestamp": time.time()
            })
            self.knowledge["active_queries"][query_id]["status"] = "answered"
            self._save_knowledge()
            logger.info(f"💡 إجابة جديدة من {agent_id} للسؤال {query_id}")
            return True
        return False

    def get_pending_queries(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """جلب الأسئلة التي تحتاج إلى إجابات (مع إبراز الأسئلة الموجهة لوكيل معين)."""
        pending = []
        for q_id, data in self.knowledge["active_queries"].items():
            if data["status"] == "open":
                q_info = {"id": q_id, **data}
                # وسم السؤال إذا كان موجهاً خصيصاً لهذا الوكيل
                if agent_id and data.get("target_expert") == agent_id:
                    q_info["priority"] = "HIGH (Direct Expert Request)"
                pending.append(q_info)
        return pending

    def check_my_answers(self, agent_id: str) -> List[Dict[str, Any]]:
        """التحقق من وجود إجابات لأسئلة وكيل معين."""
        my_queries = []
        for q_id, data in self.knowledge["active_queries"].items():
            if data["asker"] == agent_id and data["answers"]:
                my_queries.append({"id": q_id, **data})
        return my_queries

shared_experience = SharedExperienceManager()
