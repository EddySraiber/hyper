from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import json
from datetime import datetime


class ComponentBase(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"algotrading.{name}")
        self.is_running = False
        
    @abstractmethod
    def start(self) -> None:
        pass
        
    @abstractmethod
    def stop(self) -> None:
        pass
        
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass
        
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "is_running": self.is_running,
            "timestamp": datetime.utcnow().isoformat()
        }


class PersistentComponent(ComponentBase):
    def __init__(self, name: str, config: Dict[str, Any], data_dir: str = "/app/data"):
        super().__init__(name, config)
        self.data_dir = data_dir
        self.memory_file = f"{data_dir}/{name}_memory.json"
        self.memory = self._load_memory()
        
    def _load_memory(self) -> Dict[str, Any]:
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            self.logger.error(f"Error loading memory: {e}")
            return {}
            
    def _save_memory(self) -> None:
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving memory: {e}")
            
    def update_memory(self, key: str, value: Any) -> None:
        self.memory[key] = value
        self._save_memory()
        
    def get_memory(self, key: str, default: Any = None) -> Any:
        return self.memory.get(key, default)