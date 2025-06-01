import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

sys.path.append(str(Path(__file__).parent.parent))

from shared.connectors.questdb import QuestDBConnector
from shared.utils.logging import setup_logger
from services.feature_engine.indicators.technical import TechnicalIndicatorCalculator
from services.feature_engine.indicators.custom import CustomIndicatorCalculator
from services.feature_engine.indicators.microstructure import MicrostructureCalculator


def test_talib_integration():
    logger = setup_logger(__name__)
    questdb = QuestDBConnector()

    logger.info("Testing TA-Lib integration...")

    symbol = 'BTCUSDT'
    interval = '1h'

    try:
        df = questdb.get_klines_df(symbol, interval, limit=500)

        if df.empty:
            logger.error("No data found for testing")
            return

        logger.info(f"Loaded {len(df)} klines for {symbol} {interval}")

        technical_calc = TechnicalIndicatorCalculator()
        custom_calc = CustomIndicatorCalculator()
        micro_calc = MicrostructureCalculator()

        start_time = time.time()
        tech_features = technical_calc.calculate(df)
        tech_time = time.time() - start_time
        logger.info(f"Technical indicators calculated in {tech_time:.3f}s - {len(tech_features.columns)} features")

        start_time = time.time()
        custom_features = custom_calc.calculate(df)
        custom_time = time.time() - start_time
        logger.info(f"Custom indicators calculated in {custom_time:.3f}s - {len(custom_features.columns)} features")

        start_time = time.time()
        micro_features = micro_calc.calculate(df)
        micro_time = time.time() - start_time
        logger.info(f"Microstructure indicators calculated in {micro_time:.3f}s - {len(micro_features.columns)} features")

        all_features = pd.concat([tech_features, custom_features, micro_features], axis=1)
        logger.info(f"Total features generated: {len(all_features.columns)}")

        sample_features = all_features.iloc[-1]
        logger.info("\nSample feature values (latest):")
        for feature in ['sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_upper', 'atr_14',
                       'volatility_20', 'trend_strength', 'kyle_lambda']:
            if feature in sample_features:
                value = sample_features[feature]
                if not pd.isna(value):
                    logger.info(f"  {feature}: {value:.4f}")

        nan_counts = all_features.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"\nFeatures with NaN values:")
            for col, count in nan_counts[nan_counts > 0].items():
                logger.warning(f"  {col}: {count} NaNs ({count/len(all_features)*100:.1f}%)")

        logger.info("\nTA-Lib integration test successful!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        questdb.close()


if __name__ == "__main__":
    test_talib_integration()