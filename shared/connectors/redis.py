import redis
import json
from typing import Any, Dict, List, Optional
from ..utils.config import Config
from ..utils.logging import setup_logger
import pickle


class RedisConnector:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger(__name__)

        self.client = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            db=self.config.redis.db,
            decode_responses=False
        )

        self.pubsub = self.client.pubsub()

    def set(self, key: str, value: Any, expire: Optional[int] = None):
        serialized = pickle.dumps(value)
        self.client.set(key, serialized, ex=expire)

    def get(self, key: str) -> Any:
        value = self.client.get(key)
        if value:
            return pickle.loads(value)
        return None

    def set_json(self, key: str, value: Dict[str, Any], expire: Optional[int] = None):
        self.client.set(key, json.dumps(value), ex=expire)

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None

    def publish(self, channel: str, message: Any):
        if isinstance(message, dict):
            message = json.dumps(message)
        self.client.publish(channel, message)

    def subscribe(self, channels: List[str]):
        self.pubsub.subscribe(*channels)

    def get_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        message = self.pubsub.get_message(timeout=timeout)
        if message and message['type'] == 'message':
            try:
                message['data'] = json.loads(message['data'])
            except:
                pass
        return message

    def lpush(self, key: str, value: Any):
        serialized = pickle.dumps(value)
        self.client.lpush(key, serialized)

    def rpop(self, key: str) -> Any:
        value = self.client.rpop(key)
        if value:
            return pickle.loads(value)
        return None

    def llen(self, key: str) -> int:
        return self.client.llen(key)

    def delete(self, *keys):
        self.client.delete(*keys)

    def exists(self, key: str) -> bool:
        return bool(self.client.exists(key))

    def expire(self, key: str, seconds: int):
        self.client.expire(key, seconds)

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except:
            return False

    def close(self):
        self.pubsub.close()
        self.client.close()