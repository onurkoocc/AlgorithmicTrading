import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from shared.utils.config import Config
from shared.utils.logging import setup_logger
from shared.connectors.questdb import QuestDBConnector
from services.data_collector.collectors.binance import BinanceFuturesCollector


async def download_historical_data(days_back: int = 365):
    config = Config()
    logger = setup_logger(__name__)
    questdb = QuestDBConnector()

    symbols = config.symbols
    intervals = config.intervals

    collector = BinanceFuturesCollector(symbols, intervals)

    start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    end_time = int(datetime.now().timestamp() * 1000)

    logger.info(f"Downloading {days_back} days of historical data")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Intervals: {intervals}")

    for symbol in symbols:
        for interval in intervals:
            logger.info(f"Downloading {symbol} {interval}")

            try:
                klines = await collector.fetch_historical_klines(
                    symbol, interval, start_time, end_time
                )

                if klines:
                    logger.info(f"Downloaded {len(klines)} klines for {symbol} {interval}")

                    batch_size = 10000
                    for i in range(0, len(klines), batch_size):
                        batch = klines[i:i + batch_size]
                        batch_data = [kline.to_dict() for kline in batch]
                        questdb.batch_write_klines(symbol, interval, batch_data)

                    logger.info(f"Saved {len(klines)} klines to database")
                else:
                    logger.warning(f"No data received for {symbol} {interval}")

            except Exception as e:
                logger.error(f"Error downloading {symbol} {interval}: {e}")

            await asyncio.sleep(1)

    questdb.close()
    logger.info("Historical data download complete")


if __name__ == "__main__":
    days = 365
    if len(sys.argv) > 1:
        days = int(sys.argv[1])

    asyncio.run(download_historical_data(days))