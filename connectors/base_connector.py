
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseConnector(ABC):
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def connect(self, data: Dict[str, Any]) -> Dict[str, Any]: ...

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.connect(data)

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}'>"
