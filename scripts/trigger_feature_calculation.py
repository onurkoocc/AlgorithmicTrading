import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from shared.connectors.redis import RedisConnector
from shared.utils.logging import setup_logger


def trigger_feature_calculation():
    logger = setup_logger(__name__)
    redis = RedisConnector()

    logger.info("Triggering feature calculation...")

    redis.set_json('data_initialization_status', {
        'status': 'completed',
        'timestamp': time.time(),
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'intervals': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
    }, expire=86400)

    redis.publish('data_initialization', 'completed')

    logger.info("Feature calculation trigger sent")

    redis.close()


if __name__ == "__main__":
    import time

    trigger_feature_calculation()