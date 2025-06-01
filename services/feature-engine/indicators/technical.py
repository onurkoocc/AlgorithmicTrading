import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional, Dict


class TechnicalIndicatorCalculator:
    def __init__(self):
        self.indicators_config = {
            'sma': [10, 20, 50, 100, 200],
            'ema': [10, 20, 50, 100, 200],
            'rsi': [14, 21],
            'atr': [14, 21],
            'bb': {'length': 20, 'std': 2},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'adx': {'length': 14},
            'cci': {'length': 20},
            'stoch': {'k': 14, 'd': 3, 'smooth_k': 3},
            'mfi': {'length': 14},
            'williams_r': {'length': 14}
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
        for period in self.indicators_config['sma']:
            if len(df) >= period:
                features[f'sma_{period}'] = df['close'].rolling(window=period).mean()

        for period in self.indicators_config['ema']:
            if len(df) >= period:
                features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        return features

    def _calculate_momentum_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for period in self.indicators_config['rsi']:
            if len(df) > period:
                features[f'rsi_{period}'] = ta.rsi(df['close'], length=period)

        if len(df) > self.indicators_config['macd']['slow']:
            macd_config = self.indicators_config['macd']
            macd_result = ta.macd(
                df['close'],
                fast=macd_config['fast'],
                slow=macd_config['slow'],
                signal=macd_config['signal']
            )

            if macd_result is not None and not macd_result.empty:
                col_prefix = f"MACD_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"
                if f"{col_prefix}" in macd_result.columns:
                    features['macd'] = macd_result[f"{col_prefix}"]
                    features['macd_signal'] = macd_result[
                        f"MACDs_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"]
                    features['macd_hist'] = macd_result[
                        f"MACDh_{macd_config['fast']}_{macd_config['slow']}_{macd_config['signal']}"]

        if len(df) > self.indicators_config['cci']['length']:
            features['cci_20'] = ta.cci(
                df['high'], df['low'], df['close'],
                length=self.indicators_config['cci']['length']
            )

        if len(df) > self.indicators_config['stoch']['k']:
            stoch_result = ta.stoch(
                df['high'], df['low'], df['close'],
                k=self.indicators_config['stoch']['k'],
                d=self.indicators_config['stoch']['d'],
                smooth_k=self.indicators_config['stoch']['smooth_k']
            )

            if stoch_result is not None and not stoch_result.empty:
                if len(stoch_result.columns) >= 2:
                    features['stoch_k'] = stoch_result.iloc[:, 0]
                    features['stoch_d'] = stoch_result.iloc[:, 1]

        if len(df) > self.indicators_config['williams_r']['length']:
            features['williams_r'] = ta.willr(
                df['high'], df['low'], df['close'],
                length=self.indicators_config['williams_r']['length']
            )

        if len(df) > self.indicators_config['mfi']['length'] and 'volume' in df.columns:
            try:
                mfi_result = ta.mfi(
                    df['high'], df['low'], df['close'], df['volume'],
                    length=self.indicators_config['mfi']['length']
                )

                if mfi_result is not None:
                    features['mfi_14'] = mfi_result.astype(np.float64)
                else:
                    features['mfi_14'] = pd.Series(50.0, index=df.index, dtype=np.float64)
            except:
                features['mfi_14'] = pd.Series(50.0, index=df.index, dtype=np.float64)

        return features

    def _calculate_volatility_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if len(df) > self.indicators_config['bb']['length']:
            bb_result = ta.bbands(
                df['close'],
                length=self.indicators_config['bb']['length'],
                std=self.indicators_config['bb']['std']
            )

            if bb_result is not None and not bb_result.empty:
                bb_cols = bb_result.columns
                features['bb_lower'] = bb_result.iloc[:, 0]
                features['bb_middle'] = bb_result.iloc[:, 1]
                features['bb_upper'] = bb_result.iloc[:, 2]

                bb_range = features['bb_upper'] - features['bb_lower']
                features['bb_width'] = bb_range
                features['bb_percent'] = np.where(
                    bb_range > 0,
                    (df['close'] - features['bb_lower']) / bb_range,
                    0.5
                )

        for period in self.indicators_config['atr']:
            if len(df) > period:
                features[f'atr_{period}'] = ta.atr(df['high'], df['low'], df['close'], length=period)

        return features

    def _calculate_volume_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'volume' in df.columns:
            features['obv'] = ta.obv(df['close'], df['volume'])

            if len(df) > 0:
                try:
                    vwap_result = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                    if vwap_result is not None:
                        features['vwap'] = vwap_result
                except:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    cumulative_tpv = (typical_price * df['volume']).cumsum()
                    cumulative_volume = df['volume'].cumsum()
                    features['vwap'] = cumulative_tpv / cumulative_volume.replace(0, np.nan)

            if len(df) >= 20:
                volume_sma = df['volume'].rolling(window=20).mean()
                features['volume_ratio'] = df['volume'] / volume_sma.replace(0, np.nan)

        return features

    def _calculate_trend_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if len(df) > self.indicators_config['adx']['length']:
            adx_result = ta.adx(
                df['high'], df['low'], df['close'],
                length=self.indicators_config['adx']['length']
            )

            if adx_result is not None and not adx_result.empty:
                adx_cols = adx_result.columns
                if len(adx_cols) >= 3:
                    features['adx_14'] = adx_result.iloc[:, 0]
                    features['plus_di'] = adx_result.iloc[:, 1]
                    features['minus_di'] = adx_result.iloc[:, 2]

        return features