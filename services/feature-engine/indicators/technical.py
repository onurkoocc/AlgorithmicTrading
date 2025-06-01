import pandas as pd
import talib
import numpy as np
from typing import Optional, Dict


class TechnicalIndicatorCalculator:
    def __init__(self):
        self.indicators_config = {
            'sma': [10, 20, 50, 100, 200],
            'ema': [10, 20, 50, 100, 200],
            'rsi': [14, 21],
            'atr': [14, 21],
            'bb': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'adx': {'timeperiod': 14},
            'cci': {'timeperiod': 20},
            'stoch': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3},
            'mfi': {'timeperiod': 14},
            'willr': {'timeperiod': 14}
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 20:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)

        features = self._calculate_moving_averages(df, features)
        features = self._calculate_momentum_indicators(df, features)
        features = self._calculate_volatility_indicators(df, features)
        features = self._calculate_volume_indicators(df, features)
        features = self._calculate_trend_indicators(df, features)

        return features

    def _calculate_moving_averages(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        close_prices = df['close'].values

        for period in self.indicators_config['sma']:
            if len(df) >= period:
                sma = talib.SMA(close_prices, timeperiod=period)
                features[f'sma_{period}'] = sma

        for period in self.indicators_config['ema']:
            if len(df) >= period:
                ema = talib.EMA(close_prices, timeperiod=period)
                features[f'ema_{period}'] = ema

        return features

    def _calculate_momentum_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else None

        for period in self.indicators_config['rsi']:
            if len(df) > period:
                rsi = talib.RSI(close_prices, timeperiod=period)
                features[f'rsi_{period}'] = rsi

        if len(df) > self.indicators_config['macd']['slowperiod']:
            macd_config = self.indicators_config['macd']
            macd, macdsignal, macdhist = talib.MACD(
                close_prices,
                fastperiod=macd_config['fastperiod'],
                slowperiod=macd_config['slowperiod'],
                signalperiod=macd_config['signalperiod']
            )
            features['macd'] = macd
            features['macd_signal'] = macdsignal
            features['macd_hist'] = macdhist

        if len(df) > self.indicators_config['cci']['timeperiod']:
            cci = talib.CCI(
                high_prices, low_prices, close_prices,
                timeperiod=self.indicators_config['cci']['timeperiod']
            )
            features['cci_20'] = cci

        if len(df) > self.indicators_config['stoch']['fastk_period']:
            slowk, slowd = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=self.indicators_config['stoch']['fastk_period'],
                slowk_period=self.indicators_config['stoch']['slowk_period'],
                slowk_matype=0,
                slowd_period=self.indicators_config['stoch']['slowd_period'],
                slowd_matype=0
            )
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd

        if len(df) > self.indicators_config['willr']['timeperiod']:
            willr = talib.WILLR(
                high_prices, low_prices, close_prices,
                timeperiod=self.indicators_config['willr']['timeperiod']
            )
            features['williams_r'] = willr

        if len(df) > self.indicators_config['mfi']['timeperiod'] and volume is not None:
            try:
                mfi = talib.MFI(
                    high_prices, low_prices, close_prices, volume,
                    timeperiod=self.indicators_config['mfi']['timeperiod']
                )
                features['mfi_14'] = mfi
            except:
                features['mfi_14'] = pd.Series(50.0, index=df.index, dtype=np.float64)

        return features

    def _calculate_volatility_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values

        if len(df) > self.indicators_config['bb']['timeperiod']:
            bb_config = self.indicators_config['bb']
            upperband, middleband, lowerband = talib.BBANDS(
                close_prices,
                timeperiod=bb_config['timeperiod'],
                nbdevup=bb_config['nbdevup'],
                nbdevdn=bb_config['nbdevdn'],
                matype=0
            )
            features['bb_upper'] = upperband
            features['bb_middle'] = middleband
            features['bb_lower'] = lowerband

            bb_range = upperband - lowerband
            features['bb_width'] = bb_range
            features['bb_percent'] = np.where(
                bb_range > 0,
                (close_prices - lowerband) / bb_range,
                0.5
            )

        for period in self.indicators_config['atr']:
            if len(df) > period:
                atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
                features[f'atr_{period}'] = atr

        return features

    def _calculate_volume_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values

        if 'volume' in df.columns:
            volume = df['volume'].values

            obv = talib.OBV(close_prices, volume)
            features['obv'] = obv

            if len(df) > 0:
                try:
                    typical_price = (high_prices + low_prices + close_prices) / 3
                    cumulative_tpv = np.cumsum(typical_price * volume)
                    cumulative_volume = np.cumsum(volume)
                    vwap = np.where(cumulative_volume > 0, cumulative_tpv / cumulative_volume, np.nan)
                    features['vwap'] = vwap
                except:
                    features['vwap'] = np.full(len(df), np.nan)

            if len(df) >= 20:
                volume_sma = talib.SMA(volume, timeperiod=20)
                features['volume_ratio'] = np.where(volume_sma > 0, volume / volume_sma, np.nan)

        return features

    def _calculate_trend_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values

        if len(df) > self.indicators_config['adx']['timeperiod']:
            adx = talib.ADX(
                high_prices, low_prices, close_prices,
                timeperiod=self.indicators_config['adx']['timeperiod']
            )
            features['adx_14'] = adx

            plus_di = talib.PLUS_DI(
                high_prices, low_prices, close_prices,
                timeperiod=self.indicators_config['adx']['timeperiod']
            )
            features['plus_di'] = plus_di

            minus_di = talib.MINUS_DI(
                high_prices, low_prices, close_prices,
                timeperiod=self.indicators_config['adx']['timeperiod']
            )
            features['minus_di'] = minus_di

        return features