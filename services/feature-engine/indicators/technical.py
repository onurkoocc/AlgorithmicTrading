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

        macd_result = ta.macd(
            df['close'],
            fast=self.indicators_config['macd']['fast'],
            slow=self.indicators_config['macd']['slow'],
            signal=self.indicators_config['macd']['signal']
        )
        if macd_result is not None:
            features['macd'] = macd_result['MACD_12_26_9']
            features['macd_signal'] = macd_result['MACDs_12_26_9']
            features['macd_hist'] = macd_result['MACDh_12_26_9']

        features[f'cci_{self.indicators_config["cci"]["length"]}'] = ta.cci(
            df['high'], df['low'], df['close'],
            length=self.indicators_config['cci']['length']
        )

        stoch_result = ta.stoch(
            df['high'], df['low'], df['close'],
            k=self.indicators_config['stoch']['k'],
            d=self.indicators_config['stoch']['d'],
            smooth_k=self.indicators_config['stoch']['smooth_k']
        )
        if stoch_result is not None:
            features['stoch_k'] = stoch_result.iloc[:, 0]
            features['stoch_d'] = stoch_result.iloc[:, 1]

        features['williams_r'] = ta.willr(
            df['high'], df['low'], df['close'],
            length=self.indicators_config['williams_r']['length']
        )

        try:
            mfi_result = ta.mfi(
                df['high'], df['low'], df['close'], df['volume'],
                length=self.indicators_config['mfi']['length']
            )
            if mfi_result is not None and len(mfi_result) > 0:
                mfi_col_name = f'mfi_{self.indicators_config["mfi"]["length"]}'
                if isinstance(mfi_result, pd.DataFrame):
                    mfi_series = mfi_result.iloc[:, 0]
                else:
                    mfi_series = mfi_result

                features[mfi_col_name] = mfi_series.astype(np.float64)
        except Exception as e:
            features[f'mfi_{self.indicators_config["mfi"]["length"]}'] = np.nan

        return features

    def _calculate_volatility_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        bb_result = ta.bbands(
            df['close'],
            length=self.indicators_config['bb']['length'],
            std=self.indicators_config['bb']['std']
        )
        if bb_result is not None:
            features['bb_lower'] = bb_result[
                f'BBL_{self.indicators_config["bb"]["length"]}_{self.indicators_config["bb"]["std"]}.0']
            features['bb_middle'] = bb_result[
                f'BBM_{self.indicators_config["bb"]["length"]}_{self.indicators_config["bb"]["std"]}.0']
            features['bb_upper'] = bb_result[
                f'BBU_{self.indicators_config["bb"]["length"]}_{self.indicators_config["bb"]["std"]}.0']

            bb_range = features['bb_upper'] - features['bb_lower']
            features['bb_width'] = bb_range
            features['bb_percent'] = np.where(
                bb_range != 0,
                (df['close'] - features['bb_lower']) / bb_range,
                0.5
            )

        for period in self.indicators_config['atr']:
            features[f'atr_{period}'] = ta.atr(df['high'], df['low'], df['close'], length=period)

        return features

    def _calculate_volume_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['obv'] = ta.obv(df['close'], df['volume'])

        features['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

        features['volume_sma_20'] = ta.sma(df['volume'], length=20)
        features['volume_ratio'] = np.where(
            features['volume_sma_20'] > 0,
            df['volume'] / features['volume_sma_20'],
            1.0
        )

        return features

    def _calculate_trend_indicators(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        adx_result = ta.adx(
            df['high'], df['low'], df['close'],
            length=self.indicators_config['adx']['length']
        )
        if adx_result is not None:
            features[f'adx_{self.indicators_config["adx"]["length"]}'] = adx_result[
                f'ADX_{self.indicators_config["adx"]["length"]}']
            features['plus_di'] = adx_result[f'DMP_{self.indicators_config["adx"]["length"]}']
            features['minus_di'] = adx_result[f'DMN_{self.indicators_config["adx"]["length"]}']

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