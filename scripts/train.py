"""
SCRIPT: TRAINING PIPELINE
--------------------------------------------
Mô tả:
    Kịch bản huấn luyện toàn diện cho hệ thống Autoscaling.
    Tự động quét cấu hình, load dữ liệu đã chuẩn bị,
    huấn luyện song song các mô hình (Prophet, XGBoost, LSTM)
    và xuất báo cáo hiệu năng chi tiết.
"""
import os
import sys
import torch
import yaml
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Models 
from src.models import ProphetForecaster, XGBoostForecaster, LSTMForecaster

# SETUP & CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config():
    # Dùng đường dẫn tuyệt đối
    config_path = PROJECT_ROOT / "config/config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    logger.error(f"❌ Không tìm thấy config tại: {config_path}")
    sys.exit(1)

CONFIG = load_config()


# TRAINER CLASS (MANAGER)

class DataflowTrainer:
    def __init__(self):
        self.config = CONFIG
        self.data_dir = PROJECT_ROOT / "data"
        self.models_dir = PROJECT_ROOT / "saved_models" 
        self.results_dir = PROJECT_ROOT / "results"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_col = 'intensity'

    def load_prepared_data(self, interval, mode='train'):
        """Load dữ liệu đã qua Feature Engineering."""
        filename = f"prepared_{mode}_{interval}.csv"
        path = self.data_dir / filename
        
        if not path.exists():
            logger.warning(f"⚠️ Không tìm thấy file: {path}")
            return None
        
        df = pd.read_csv(path)
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            # Sắp xếp theo thời gian để chắc chắn
            df = df.sort_values('ds')
            
            # Xóa dòng trùng (giữ dòng cuối cùng hoặc đầu tiên)
            # keep='last' để ưu tiên dữ liệu mới nhất nếu có cập nhật
            initial_len = len(df)
            df = df.drop_duplicates(subset=['ds'], keep='last')
            
            if len(df) < initial_len:
                logger.warning(f"   🧹 Đã xóa {initial_len - len(df)} dòng trùng lặp timestamp trong {filename}")
      

        # Đảm bảo không còn NaN
        df = df.dropna()
        return df

    def train_interval(self, interval):
        """Huấn luyện tất cả mô hình cho một khung thời gian cụ thể (vd: 5min)."""
        logger.info(f"\n{'='*60}\n 🚀 BẮT ĐẦU HUẤN LUYỆN INTERVAL: {interval}\n{'='*60}")
        
        # 1. Load Data
        df_train = self.load_prepared_data(interval, 'train')
        df_test = self.load_prepared_data(interval, 'test') 
        
        if df_train is None or df_test is None:
            return

        # Tự động lấy tất cả các cột trừ cột thời gian và target
        exclude_cols = ['ds', 'timestamp', 'y', self.target_col]
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]
        logger.info(f"   🔍 Detected {len(feature_cols)} features: {feature_cols}")

        # 2. Chuẩn bị dữ liệu Matrix (X, y)
        X_train = df_train[feature_cols].values
        y_train = df_train[self.target_col].values
        X_test = df_test[feature_cols].values
        y_test = df_test[self.target_col].values

        # 3. Scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_sc = scaler_X.fit_transform(X_train)
        y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_test_sc = scaler_X.transform(X_test)
        y_test_sc = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Lưu Scaler
        joblib.dump(scaler_X, self.models_dir / f"scaler_X_{interval}.pkl")
        joblib.dump(scaler_y, self.models_dir / f"scaler_y_{interval}.pkl")

        
        # MODEL 1: PROPHET
        
        if self.config['models']['prophet']['enabled']:
            model = ProphetForecaster(self.config)
            
            # Tạo bản sao để không ảnh hưởng dữ liệu gốc
            pf_train = df_train.copy()
            
            # Nếu trong file có cột y (cột target cũ), xóa nó đi
            if 'y' in pf_train.columns:
                pf_train = pf_train.drop(columns=['y'])
            
            # Rename cột cho đúng định dạng Prophet
            pf_train = pf_train.rename(columns={'ds': 'ds', self.target_col: 'y'})
            
            # Reset index để đảm bảo an toàn tuyệt đối
            pf_train = pf_train.reset_index(drop=True)
            
            # Chọn Regressors
            pf_regressors = [c for c in feature_cols if c in pf_train.columns]
            
            model.fit(pf_train, regressors=pf_regressors)
            model.save(self.models_dir / f"prophet_{interval}.pkl")

       
        # MODEL 2: XGBOOST
        
        if self.config['models']['xgboost']['enabled']:
            model = XGBoostForecaster(self.config)
            # XGBoost dùng dữ liệu đã Scaling hoặc Raw đều được
            model.fit(X_train_sc, y_train_sc, X_test_sc, y_test_sc)
            
            # Eval sơ bộ
            preds = model.predict(X_test_sc)
            metrics = model.evaluate(y_test_sc, preds)
            logger.info(f"📊 XGBoost Results ({interval}): {metrics}")
            
            model.save(self.models_dir / f"xgboost_{interval}.pkl")

        
        # MODEL 3: LSTM
       
        if self.config['models']['lstm']['enabled']:
            input_dim = X_train_sc.shape[1]
            model = LSTMForecaster(self.config, input_dim)      

            model.fit(X_train_sc, y_train_sc, X_test_sc, y_test_sc)
            
            # Lưu model Pytorch 
            torch.save(model.model.state_dict(), self.models_dir / f"lstm_{interval}.pth")
            logger.info(f"✅ LSTM Model saved to lstm_{interval}.pth")

    def run_complete_training(self):
        """Chạy toàn bộ pipeline cho mọi interval."""
        intervals = self.config['processing']['intervals']
        for interval in intervals:
            try:
                self.train_interval(interval)
            except Exception as e:
                logger.error(f"❌ Lỗi khi train interval {interval}: {e}")
                import traceback
                traceback.print_exc()

# MAIN ENTRY

if __name__ == "__main__":
    trainer = DataflowTrainer()
    trainer.run_complete_training()
    
    print("\n" + "="*60)
    print("🎉 HUẤN LUYỆN HOÀN TẤT! MODEL ĐÃ SẴN SÀNG TRONG THƯ MỤC 'saved_models/'")
    print("="*60)