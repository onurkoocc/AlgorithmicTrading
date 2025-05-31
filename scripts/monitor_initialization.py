import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from shared.connectors.redis import RedisConnector
from shared.utils.logging import setup_logger


def monitor_initialization():
    logger = setup_logger(__name__)
    redis = RedisConnector()

    logger.info("Monitoring data initialization and feature calculation progress...")

    start_time = time.time()
    data_initialized = False

    while True:
        try:
            init_status = redis.get_json('data_initialization_status')
            if init_status:
                if not data_initialized and init_status.get('status') == 'completed':
                    data_initialized = True
                    init_time = datetime.fromtimestamp(init_status['timestamp'])
                    logger.info(f"Data initialization completed at {init_time}")
                    logger.info(f"Symbols: {init_status['symbols']}")
                    logger.info(f"Intervals: {init_status['intervals']}")

            collector_health = redis.get_json('health:data-collector')
            if collector_health:
                logger.info(f"Data Collector Status: {collector_health['status']}")
                logger.info(f"  Active streams: {collector_health['active_streams']}")
                logger.info(f"  Buffer size: {collector_health['buffer_size']}")
                logger.info(f"  Data initialized: {collector_health['data_initialized']}")

            feature_progress = {}
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                for interval in ['1h', '1d']:
                    quality_key = f"quality:{symbol}:{interval}"
                    quality_data = redis.get_json(quality_key)
                    if quality_data:
                        feature_progress[f"{symbol}:{interval}"] = {
                            'quality_score': quality_data['overall_quality_score'],
                            'total_features': quality_data['total_features']
                        }

            if feature_progress:
                logger.info("Feature calculation progress:")
                for key, data in feature_progress.items():
                    logger.info(
                        f"  {key}: {data['total_features']} features, quality score: {data['quality_score']:.2f}")

            elapsed = time.time() - start_time
            logger.info(f"Elapsed time: {elapsed / 60:.1f} minutes")
            logger.info("-" * 50)

            time.sleep(10)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(5)

    redis.close()
    logger.info("Monitoring stopped")


if __name__ == "__main__":
    monitor_initialization()