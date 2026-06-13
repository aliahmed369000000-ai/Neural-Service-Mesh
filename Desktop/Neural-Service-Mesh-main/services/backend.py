from __future__ import annotations

"""Shared backend provider for trainer/CKG singletons.
This keeps a single KnowledgeTrainer instance for the dashboard to call directly.
"""

from ai.knowledge_trainer import KnowledgeTrainer

_trainer = None

def get_trainer() -> KnowledgeTrainer:
    global _trainer
    if _trainer is None:
        _trainer = KnowledgeTrainer()
    return _trainer
