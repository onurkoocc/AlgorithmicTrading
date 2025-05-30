from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
from queue import Queue
import threading


class BaseCollector(ABC):
    def __init__(self, symbols: List[str], intervals: List[str]):
        self.symbols = symbols
        self.intervals = intervals
        self.is_running = False
        self.data_queue = Queue()

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def disconnect(self):
        pass

    @abstractmethod
    async def subscribe(self, symbol: str, interval: str):
        pass

    @abstractmethod
    async def unsubscribe(self, symbol: str, interval: str):
        pass

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    def get_data(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        try:
            return self.data_queue.get(timeout=timeout)
        except:
            return None