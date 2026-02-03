"""
SRC FEATURE ENGINEERING
--------------------------------
Mô tả: Chứa Class xử lý Feature Engineering.
"""
import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    def __init__(self, config):
        self.config = config
        
    def _parse_interval_minutes(self, interval_str):
        match = re.match(r"(\d+)", interval_str)
        return int(match.group(1)) if match else 5

    def _restore_time_continuity(self, df, interval_str):
        if 'ds' not in df.columns: raise ValueError("Thiếu cột 'ds'")
        df = df.set_index('ds').sort_index()
        minutes = self._parse_interval_minutes(interval_str)
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f"{minutes}min")
        df_restored = df.reindex(full_idx)
        return df_restored.reset_index().rename(columns={'index': 'ds'})

    def generate_cyclical_features(self, df):
        df = df.copy()
        df['hour'] = df['ds'].dt.hour
        df['day_of_week'] = df['ds'].dt.dayofweek
        
        # 1. Tạo features (5 cột)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 2. Xóa cột thô 
        df.drop(columns=['hour', 'day_of_week'], inplace=True)
        return df

    def generate_lag_rolling_features(self, df, interval_str):
        df = df.copy()
        target_col = 'intensity'
        minutes = self._parse_interval_minutes(interval_str)
        steps_per_hour = 60 // minutes
        
        # A. LAGS (4 Features)
        lags = {
            'lag_1step': 1,
            'lag_1h': steps_per_hour,
            'lag_24h': 24 * steps_per_hour,
            'lag_7d': 7 * 24 * steps_per_hour
        }
        for name, step in lags.items():
            df[name] = df[target_col].shift(step)
            
        # B. ROLLING (3 Features)
        window_size = steps_per_hour * 4 
        shifted = df[target_col].shift(1)
        df['roll_mean_4h'] = shifted.rolling(window=window_size).mean()
        df['roll_std_4h'] = shifted.rolling(window=window_size).std()
        df['roll_max_4h'] = shifted.rolling(window=window_size).max()
        return df

    def cleanup_and_validate(self, df, target_col='intensity'):
        df_clean = df.dropna(subset=[target_col, 'lag_24h', 'roll_mean_4h']).copy()
        return df_clean.fillna(0)

    def process(self, df, interval, context_df=None):
        """Hàm xử lý chính cho 1 dataframe"""
        # 1. Chỉ giữ 2 cột cần thiết
        df = df[['ds', 'intensity']].copy()
        
        # 2. Nối Context (nếu là Test)
        original_start = None
        if context_df is not None:
            original_start = df['ds'].min()
            df = pd.concat([context_df, df], axis=0, ignore_index=True)
            df = df.drop_duplicates(subset=['ds']).sort_values('ds')

        # 3. Feature Engineering
        df = self._restore_time_continuity(df, interval)
        df = self.generate_cyclical_features(df)
        df = self.generate_lag_rolling_features(df, interval)
        
        # 4. Cắt về kích thước cũ (nếu là Test)
        if original_start is not None:
            df = df[df['ds'] >= original_start].copy()
            
        # 5. Clean
        df = self.cleanup_and_validate(df)
        return df