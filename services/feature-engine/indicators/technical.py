import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional


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
        features = pd.DataFrame(index=df.index)

        features = self._calculate_moving_averages(df, features)
        features = self._calculate_momentum_indicators(df, features)
        features = self._calculate_volatility_indicators(df, features)
        features = self._calculate_volume_indicators(df, features)
        features = self._calculate_trend_indicators(df, features)

        return features

    def _calculate_moving_averages(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for period in self.indicators_config['sma']:
            features[f'sma_{period}'] = ta.sma(df['close'], length=period)

        for period in self.indicators_config['ema']:
            features[f'ema_{period}'] = ta.ema(df['close'], length=period)

        return features

    def _calculate_momentum_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for period in self.indicators_config['rsi']:
            features[f'rsi_{period}'] = ta.rsi(df['close'], length=period)

        macd_config = self.indicators_config['macd']
        macd_result = ta.macd(
            df['close'],
            fast=macd_config['fast'],
            slow=macd_config['slow'],
            signal=macd_config['signal']
        )
        if macd_result is not None and not macd_result.empty:
            features[f'macd'] = macd_result[f'MACD_{macd_config["fast"]}_{macd_config["slow"]}_{macd_config["signal"]}']
            features[f'macd_signal'] = macd_result[f'MACDs_{macd_config["fast"]}_{macd_config["slow"]}_{macd_config["signal"]}']
            features[f'macd_hist'] = macd_result[f'MACDh_{macd_config["fast"]}_{macd_config["slow"]}_{macd_config["signal"]}']

        cci_length = self.indicators_config['cci']['length']
        features[f'cci_{cci_length}'] = ta.cci(
            df['high'], df['low'], df['close'],
            length=cci_length
        )

        stoch_config = self.indicators_config['stoch']
        stoch_result = ta.stoch(
            df['high'], df['low'], df['close'],
            k=stoch_config['k'],
            d=stoch_config['d'],
            smooth_k=stoch_config['smooth_k']
        )
        if stoch_result is not None and not stoch_result.empty:
            k_col = f'STOCHk_{stoch_config["k"]}_{stoch_config["d"]}_{stoch_config["smooth_k"]}'
            d_col = f'STOCHd_{stoch_config["k"]}_{stoch_config["d"]}_{stoch_config["smooth_k"]}'
            if k_col in stoch_result.columns and d_col in stoch_result.columns:
                features['stoch_k'] = stoch_result[k_col]
                features['stoch_d'] = stoch_result[d_col]
            elif len(stoch_result.columns) >= 2:
                features['stoch_k'] = stoch_result.iloc[:, 0]
                features['stoch_d'] = stoch_result.iloc[:, 1]

        willr_length = self.indicators_config['williams_r']['length']
        features['williams_r'] = ta.willr(
            df['high'], df['low'], df['close'],
            length=willr_length
        )

        mfi_length = self.indicators_config['mfi']['length']
        mfi_col_name = f'mfi_{mfi_length}'
        try:
            high_s = df['high'].astype(np.float64)
            low_s = df['low'].astype(np.float64)
            close_s = df['close'].astype(np.float64)
            volume_s = df['volume'].fillna(0).astype(np.float64)
            volume_for_mfi = volume_s / 1e6

            mfi_result = ta.mfi(
                high=high_s, low=low_s, close=close_s, volume=volume_for_mfi,
                length=mfi_length
            )

            if mfi_result is not None and not mfi_result.empty:
                if isinstance(mfi_result, pd.Series):
                    features[mfi_col_name] = mfi_result.astype(np.float64).fillna(50.0).clip(lower=0.0, upper=100.0)
                else:
                    features[mfi_col_name] = pd.Series(mfi_result, index=df.index, dtype=np.float64).fillna(50.0).clip(lower=0.0, upper=100.0)
            else:
                features[mfi_col_name] = pd.Series(50.0, index=df.index, dtype=np.float64)

        except Exception as e:
            features[mfi_col_name] = pd.Series(50.0, index=df.index, dtype=np.float64)

        return features

    def _calculate_volatility_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        bb_config = self.indicators_config['bb']
        bb_result = ta.bbands(
            df['close'],
            length=bb_config['length'],
            std=bb_config['std']
        )
        if bb_result is not None and not bb_result.empty:
            bb_lower_col = f'BBL_{bb_config["length"]}_{bb_config["std"]}.0'
            bb_middle_col = f'BBM_{bb_config["length"]}_{bb_config["std"]}.0'
            bb_upper_col = f'BBU_{bb_config["length"]}_{bb_config["std"]}.0'

            features['bb_lower'] = bb_result[bb_lower_col]
            features['bb_middle'] = bb_result[bb_middle_col]
            features['bb_upper'] = bb_result[bb_upper_col]

            bb_range = features['bb_upper'] - features['bb_lower']
            features['bb_width'] = bb_range.fillna(0)
            features['bb_percent'] = np.where(
                bb_range.fillna(0) != 0,
                (df['close'] - features['bb_lower']) / bb_range,
                0.5
            )
            features['bb_percent'] = features['bb_percent'].fillna(0.5)

        for period in self.indicators_config['atr']:
            features[f'atr_{period}'] = ta.atr(df['high'], df['low'], df['close'], length=period)

        return features

    def _calculate_volume_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['obv'] = ta.obv(df['close'], df['volume'].fillna(0).astype(np.float64))
        features['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'].fillna(0).astype(np.float64))

        volume_sma_20 = ta.sma(df['volume'].fillna(0).astype(np.float64), length=20)
        features['volume_sma_20'] = volume_sma_20
        features['volume_ratio'] = np.where(
            volume_sma_20.fillna(0) > 0,
            df['volume'].fillna(0).astype(np.float64) / volume_sma_20,
            1.0
        )
        features['volume_ratio'] = features['volume_ratio'].fillna(1.0)

        return features

    def _calculate_trend_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        adx_length = self.indicators_config['adx']['length']
        adx_result = ta.adx(
            df['high'], df['low'], df['close'],
            length=adx_length
        )
        if adx_result is not None and not adx_result.empty:
            features[f'adx_{adx_length}'] = adx_result[f'ADX_{adx_length}']
            features['plus_di'] = adx_result[f'DMP_{adx_length}']
            features['minus_di'] = adx_result[f'DMN_{adx_length}']

        pivot_result = self._calculate_pivot_points(df)
        for key, value in pivot_result.items():
            features[key] = value

        return features

    def _calculate_pivot_points(self, df: pd.DataFrame) -> dict:
        high = df['high'].shift(1)
        low = df['low'].shift(1)
        close = df['close'].shift(1)

        pivot = (high + low + close) / 3

        pivot_points = {
            'pivot': pivot,
            'resistance_1': 2 * pivot - low,
            'resistance_2': pivot + (high - low),
            'resistance_3': high + 2 * (pivot - low),
            'support_1': 2 * pivot - high,
            'support_2': pivot - (high - low),
            'support_3': low - 2 * (high - pivot)
        }

        for key, value in pivot_points.items():
            pivot_points[key] = value.ffill().fillna(0)

        return pivot_points