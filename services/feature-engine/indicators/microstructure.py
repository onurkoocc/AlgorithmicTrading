import pandas as pd
import numpy as np
from typing import Optional


class MicrostructureCalculator:
    def __init__(self):
        self.lookback_windows = [5, 10, 20, 50]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        features = self._calculate_spread_features(df, features)
        features = self._calculate_order_flow_features(df, features)
        features = self._calculate_volume_profile_features(df, features)
        features = self._calculate_price_impact_features(df, features)
        features = self._calculate_liquidity_features(df, features)

        return features

    def _calculate_spread_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['avg_spread'] = (df['high'] - df['low']).rolling(window=20).mean()
        features['spread_volatility'] = (df['high'] - df['low']).rolling(window=20).std()

        relative_spread = pd.Series(0.0, index=df.index)
        close_mask = df['close'] > 0
        relative_spread[close_mask] = (df['high'][close_mask] - df['low'][close_mask]) / df['close'][close_mask]
        features['relative_spread'] = relative_spread

        avg_spread_shifted = features['avg_spread'].shift(10)
        spread_momentum = pd.Series(1.0, index=df.index)
        spread_mask = avg_spread_shifted > 0
        spread_momentum[spread_mask] = features['avg_spread'][spread_mask] / avg_spread_shifted[spread_mask]
        features['spread_momentum'] = spread_momentum

        return features

    def _calculate_order_flow_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'taker_buy_volume' in df.columns:
            buy_pressure = pd.Series(0.5, index=df.index)
            vol_mask = df['volume'] > 0
            buy_pressure[vol_mask] = df['taker_buy_volume'][vol_mask] / df['volume'][vol_mask]
            features['buy_pressure'] = buy_pressure
            features['sell_pressure'] = 1 - features['buy_pressure']

            features['order_flow_imbalance'] = features['buy_pressure'].rolling(window=20).mean() - 0.5

            close_volume = (df['volume'] * df['close']).rolling(window=20).sum()
            taker_buy_value = (df['taker_buy_volume'] * df['close']).rolling(window=20).sum()

            volume_weighted_buy_pressure = pd.Series(0.5, index=df.index)
            cv_mask = close_volume > 0
            volume_weighted_buy_pressure[cv_mask] = taker_buy_value[cv_mask] / close_volume[cv_mask]
            features['volume_weighted_buy_pressure'] = volume_weighted_buy_pressure

        return features

    def _calculate_volume_profile_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for window in self.lookback_windows:
            close_volume_sum = (df['close'] * df['volume']).rolling(window=window).sum()
            volume_sum = df['volume'].rolling(window=window).sum()

            vwap = pd.Series(np.nan, index=df.index)
            vol_sum_mask = volume_sum > 0
            vwap[vol_sum_mask] = close_volume_sum[vol_sum_mask] / volume_sum[vol_sum_mask]

            vwap_deviation = pd.Series(0.0, index=df.index)
            vwap_mask = vwap > 0
            vwap_deviation[vwap_mask] = (df['close'][vwap_mask] - vwap[vwap_mask]) / vwap[vwap_mask]
            features[f'vwap_deviation_{window}'] = vwap_deviation

        volume_5_sum = df['volume'].rolling(window=5).sum()
        volume_20_sum = df['volume'].rolling(window=20).sum()

        volume_concentration = pd.Series(1.0, index=df.index)
        vol_20_mask = volume_20_sum > 0
        volume_concentration[vol_20_mask] = volume_5_sum[vol_20_mask] / volume_20_sum[vol_20_mask]
        features['volume_concentration'] = volume_concentration

        if 'trades' in df.columns:
            avg_trade_size = pd.Series(0.0, index=df.index)
            trades_mask = df['trades'] > 0
            avg_trade_size[trades_mask] = df['volume'][trades_mask] / df['trades'][trades_mask]
            features['avg_trade_size'] = avg_trade_size

            avg_trade_size_shifted = features['avg_trade_size'].shift(20)
            trade_size_momentum = pd.Series(1.0, index=df.index)
            ats_mask = avg_trade_size_shifted > 0
            trade_size_momentum[ats_mask] = features['avg_trade_size'][ats_mask] / avg_trade_size_shifted[ats_mask]
            features['trade_size_momentum'] = trade_size_momentum

        return features

    def _calculate_price_impact_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        volume_bars = df['volume'].rolling(window=20).mean()
        price_change = np.abs(df['close'].pct_change())

        volume_ratio = pd.Series(1.0, index=df.index)
        vb_mask = volume_bars > 0
        volume_ratio[vb_mask] = df['volume'][vb_mask] / volume_bars[vb_mask]

        price_impact = pd.Series(0.0, index=df.index)
        vr_mask = volume_ratio > 1e-10
        price_impact[vr_mask] = price_change[vr_mask] / volume_ratio[vr_mask]
        features['price_impact'] = price_impact

        kyle_lambda = pd.Series(0.0, index=df.index)
        sqrt_vol = np.sqrt(df['volume'])
        sv_mask = sqrt_vol > 1e-10
        kyle_lambda[sv_mask] = np.abs(df['close'].diff()[sv_mask]) / sqrt_vol[sv_mask]
        features['kyle_lambda'] = kyle_lambda

        close_volume = df['volume'] * df['close']
        amihud_illiquidity = pd.Series(0.0, index=df.index)
        cv_mask = close_volume > 1e-10
        amihud_illiquidity[cv_mask] = np.abs(df['close'].pct_change()[cv_mask]) / close_volume[cv_mask]
        features['amihud_illiquidity'] = amihud_illiquidity.rolling(window=20).mean()

        return features

    def _calculate_liquidity_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        volume_20_sum = df['volume'].rolling(window=20).sum()
        volume_100_sum = df['volume'].rolling(window=100).sum()

        volume_turnover = pd.Series(1.0, index=df.index)
        v100_mask = volume_100_sum > 0
        volume_turnover[v100_mask] = volume_20_sum[v100_mask] / volume_100_sum[v100_mask]
        features['volume_turnover'] = volume_turnover

        high_low_spread = df['high'] - df['low']
        features['effective_spread'] = 2 * np.abs(df['close'] - (df['high'] + df['low']) / 2)

        high_low_ratio = pd.Series(1.0, index=df.index)
        low_mask = df['low'] > 0
        high_low_ratio[low_mask] = df['high'][low_mask] / df['low'][low_mask]

        log_hl_ratio = pd.Series(0.0, index=df.index)
        hl_mask = high_low_ratio > 0
        log_hl_ratio[hl_mask] = np.log(high_low_ratio[hl_mask])

        features['realized_volatility'] = np.sqrt(
            (log_hl_ratio ** 2 / (4 * np.log(2))).rolling(window=20).mean()
        )

        if 'quote_volume' in df.columns:
            qv_5_mean = df['quote_volume'].rolling(window=5).mean()
            qv_20_mean = df['quote_volume'].rolling(window=20).mean()

            depth_imbalance = pd.Series(1.0, index=df.index)
            qv20_mask = qv_20_mean > 0
            depth_imbalance[qv20_mask] = qv_5_mean[qv20_mask] / qv_20_mean[qv20_mask]
            features['depth_imbalance'] = depth_imbalance

        return features