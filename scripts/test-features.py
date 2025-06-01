import sys
from pathlib import Path
import pandas as pd
import time

sys.path.append(str(Path(__file__).parent.parent))

from shared.connectors.questdb import QuestDBConnector
from shared.connectors.redis import RedisConnector
from shared.utils.logging import setup_logger


def test_feature_pipeline():
    logger = setup_logger(__name__)
    questdb = QuestDBConnector()
    redis = RedisConnector()

    logger.info("Testing feature engineering pipeline...")

    symbols = ['BTCUSDT', 'ETHUSDT']
    intervals = ['1h', '4h', '1d']

    for symbol in symbols:
        for interval in intervals:
            try:
                klines_count = questdb.verify_data(symbol, interval)
                logger.info(f"Klines for {symbol} {interval}: {klines_count}")

                features_table = f"features_{interval}"
                query = f"SELECT count() as cnt FROM {features_table} WHERE symbol = '{symbol}'"
                result = questdb.execute_query(query)

                if result:
                    features_count = result[0]['cnt']
                    logger.info(f"Features for {symbol} {interval}: {features_count}")

                    if features_count > 0:
                        latest_query = f"""
                            SELECT timestamp, close, sma_20, ema_20, rsi_14, volume_ratio
                            FROM {features_table}
                            WHERE symbol = '{symbol}'
                            ORDER BY timestamp DESC
                            LIMIT 5
                        """
                        latest_features = questdb.execute_query(latest_query)

                        if latest_features:
                            logger.info(f"Latest features for {symbol} {interval}:")
                            for row in latest_features:
                                logger.info(f"  {row['timestamp']}: close={row['close']:.2f}, "
                                            f"sma_20={row.get('sma_20', 0):.2f}, rsi_14={row.get('rsi_14', 0):.2f}")

                quality_key = f"quality:{symbol}:{interval}"
                quality_data = redis.get_json(quality_key)

                if quality_data:
                    logger.info(f"Quality score for {symbol} {interval}: "
                                f"{quality_data['overall_quality_score']:.2f}")

                    if quality_data['recommendations']:
                        logger.warning(f"Recommendations: {quality_data['recommendations']}")

            except Exception as e:
                logger.error(f"Error testing {symbol} {interval}: {e}")

    messages_processed = 0
    start_time = time.time()
    timeout = 30

    logger.info(f"Monitoring real-time feature updates for {timeout} seconds...")

    channels = [f"features:{symbol}:{interval}" for symbol in symbols for interval in intervals]
    redis.subscribe(channels)

    while time.time() - start_time < timeout:
        message = redis.get_message(timeout=1.0)
        if message and message['type'] == 'message':
            messages_processed += 1
            channel = message['channel'].decode('utf-8')
            logger.info(f"Received feature update on {channel}")

    logger.info(f"Processed {messages_processed} feature updates in {timeout} seconds")

    questdb.close()
    redis.close()

    logger.info("Feature pipeline test complete!")


if __name__ == "__main__":
    test_feature_pipeline()