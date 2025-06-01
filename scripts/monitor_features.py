import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from shared.connectors.redis import RedisConnector
from shared.connectors.questdb import QuestDBConnector
from shared.utils.logging import setup_logger


def monitor_feature_progress():
    logger = setup_logger(__name__)
    redis = RedisConnector()
    questdb = QuestDBConnector()

    logger.info("Monitoring feature engineering progress...")

    symbols = ['BTCUSDT', 'ETHUSDT']
    intervals = ['1m', '5m', '15m', '1h', '4h', '1d']

    while True:
        try:
            print("\n" + "=" * 80)
            print(f"Feature Engineering Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            total_features = 0

            for symbol in symbols:
                print(f"\n{symbol}:")
                print("-" * 40)

                for interval in intervals:
                    try:
                        klines_table = f"klines_{interval}"
                        features_table = f"features_{interval}"

                        klines_query = f"SELECT count() as cnt FROM {klines_table} WHERE symbol = '{symbol}'"
                        features_query = f"SELECT count() as cnt FROM {features_table} WHERE symbol = '{symbol}'"

                        klines_result = questdb.execute_query(klines_query)
                        features_result = questdb.execute_query(features_query)

                        klines_count = klines_result[0]['cnt'] if klines_result else 0
                        features_count = features_result[0]['cnt'] if features_result else 0

                        total_features += features_count

                        quality_key = f"quality:{symbol}:{interval}"
                        quality_data = redis.get_json(quality_key)
                        quality_score = quality_data.get('overall_quality_score', 0) if quality_data else 0

                        status = "✓" if features_count > 0 else "✗"
                        progress = (features_count / klines_count * 100) if klines_count > 0 else 0

                        print(f"  {interval:>3s}: {status} Klines: {klines_count:>6} | "
                              f"Features: {features_count:>6} ({progress:>5.1f}%) | "
                              f"Quality: {quality_score:>5.1f}")

                    except Exception as e:
                        print(f"  {interval:>3s}: ✗ Error: {str(e)[:50]}...")

            print(f"\nTotal Features Generated: {total_features:,}")

            health_data = redis.get_json('health:data-collector')
            if health_data:
                print(f"\nData Collector: {health_data.get('status', 'unknown')}")
                print(f"Active Streams: {health_data.get('active_streams', 0)}")

            time.sleep(30)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(5)

    questdb.close()
    redis.close()
    logger.info("Monitoring stopped")


if __name__ == "__main__":
    monitor_feature_progress()