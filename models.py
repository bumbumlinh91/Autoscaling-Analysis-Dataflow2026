"""
MODULE: MODEL DEFINITIONS 
---------------------------------------------------
Mô tả:
    Định nghĩa các lớp mô hình dự báo theo chuẩn hướng đối tượng.
    Tất cả các mô hình kế thừa từ BaseForecaster để đảm bảo tính nhất quán trong Pipeline.

Hỗ trợ:
    1. ProphetForecaster: Mô hình thống kê (Facebook Prophet).
    2. XGBoostForecaster: Mô hình cây quyết định tăng cường (Gradient Boosting).
    3. LSTMForecaster: Mạng nơ-ron hồi quy (Deep Learning - PyTorch).

"""
import abc
import os
import joblib
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Thiết lập Logger
logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# BASE CLASS 

class BaseForecaster(abc.ABC):
    """Lớp cơ sở cho tất cả các mô hình dự báo."""
    
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.model = None
        self.features = None

    @abc.abstractmethod
    def fit(self, X, y=None):
        """Huấn luyện mô hình."""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Dự báo giá trị."""
        pass

    def save(self, path):
        """Lưu mô hình xuống đĩa."""
        logger.info(f"💾 Saving model {self.name} to {path}")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Tải mô hình từ đĩa."""
        logger.info(f"📂 Loading model from {path}")
        return joblib.load(path)

    def evaluate(self, y_true, y_pred):
        """Đánh giá hiệu năng mô hình."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return {'rmse': rmse, 'mae': mae}


# 1. PROPHET FORECASTER

class ProphetForecaster(BaseForecaster):
    def __init__(self, config):
        super().__init__("Prophet", config)
        self.params = config['models']['prophet']['params']
        self.regressors = [] # Các biến phụ trợ (Lags, Rolling)

    def fit(self, df_train, regressors=None):
        """
        Prophet yêu cầu DataFrame có cột 'ds' và 'y'.
        regressors: Danh sách các cột feature bổ sung (Lag, Rolling...).
        """
        logger.info(f"🔮 Initializing Prophet with params: {self.params}")
        
        # Khởi tạo model với tham số từ Config
        self.model = Prophet(
            growth=self.params.get('growth', 'linear'),
            seasonality_mode=self.params.get('seasonality_mode', 'additive'),
            daily_seasonality=self.params.get('daily_seasonality', True),
            weekly_seasonality=self.params.get('weekly_seasonality', True),
            yearly_seasonality=self.params.get('yearly_seasonality', False),
            changepoint_prior_scale=self.params.get('changepoint_prior_scale', 0.05),
            interval_width=self.params.get('interval_width', 0.95)
        )
        
        # Thêm Regressors 
        if regressors:
            self.regressors = regressors
            for reg in self.regressors:
                self.model.add_regressor(reg)
                
        # Fit
        self.model.fit(df_train)
        logger.info("✅ Prophet training completed.")

    def predict(self, df_future):
        """Dự báo."""
        forecast = self.model.predict(df_future)
        return forecast['yhat'].values

# 2. XGBOOST FORECASTER

class XGBoostForecaster(BaseForecaster):
    def __init__(self, config):
        super().__init__("XGBoost", config)
        self.params = config['models']['xgboost']['params']
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"🌳 Initializing XGBoost with params: {self.params}")
        
        self.model = XGBRegressor(
            n_estimators=self.params.get('n_estimators', 500),
            max_depth=self.params.get('max_depth', 7),
            learning_rate=self.params.get('learning_rate', 0.01),
            subsample=self.params.get('subsample', 0.8),
            colsample_bytree=self.params.get('colsample_bytree', 0.8),
            objective=self.params.get('objective', 'reg:squarederror'),
            random_state=self.params.get('random_state', 42),
            n_jobs=-1,
            early_stopping_rounds=50 
        )
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"✅ XGBoost trained. Best iteration: {self.model.best_iteration}")

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        return self.model.feature_importances_


# 3. LSTM FORECASTER 

# Định nghĩa kiến trúc mạng
class LSTMModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (Batch, Seq, Feature)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :] 
        return self.fc(last_step)

class LSTMForecaster(BaseForecaster):
    def __init__(self, config, input_dim):
        super().__init__("LSTM", config)
        self.params = config['models']['lstm']['params']
        self.input_dim = input_dim
        
        self.model = LSTMModule(
            input_dim=input_dim,
            hidden_dim=self.params.get('units', 64),
            num_layers=2,
            dropout=self.params.get('dropout', 0.2)
        ).to(DEVICE)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.params.get('learning_rate', 0.001)
        )
        self.criterion = nn.MSELoss()

    def create_sequences(self, X, y, seq_length):
        """Tạo Sliding Window cho LSTM."""
        xs, ys = [], []
        for i in range(len(X) - seq_length):
            xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(xs), np.array(ys)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        epochs = self.params.get('epochs', 50)
        batch_size = self.params.get('batch_size', 64)
        seq_length = self.config['models']['lstm'].get('n_lags', 30) # Lấy từ config
        
        logger.info(f"🧠 Training LSTM (Dev: {DEVICE}) | Epochs: {epochs} | Batch: {batch_size} | Seq: {seq_length}")
        
        # 1. Tạo Sequences
        X_seq_train, y_seq_train = self.create_sequences(X_train, y_train, seq_length)
        X_seq_val, y_seq_val = self.create_sequences(X_val, y_val, seq_length)
        
        # 2. DataLoader
        train_data = TensorDataset(torch.from_numpy(X_seq_train).float(), torch.from_numpy(y_seq_train).float())
        val_data = TensorDataset(torch.from_numpy(X_seq_val).float(), torch.from_numpy(y_seq_val).float())
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # 3. Training Loop với Early Stopping
        best_val_loss = float('inf')
        patience = self.params['early_stopping']['patience']
        counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(X_b)
                loss = self.criterion(pred, y_b.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                    pred = self.model(X_b)
                    loss = self.criterion(pred, y_b.view(-1, 1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Log mỗi 5 epoch
            if (epoch+1) % 5 == 0:
                logger.info(f"   Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.6f}")
                
            # Early Stopping Check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                # Lưu best state 
                best_state = self.model.state_dict()
            else:
                counter += 1
                if self.params['early_stopping']['enabled'] and counter >= patience:
                    logger.info(f"   🛑 Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(best_state) 
                    break
        
        # Đảm bảo model giữ trọng số tốt nhất
        if 'best_state' in locals():
            self.model.load_state_dict(best_state)

    def predict(self, X):
        """Dự báo với LSTM (cần xử lý Sequence)."""
        # Lưu ý: Hàm này đang giả định X đã được sequence hóa hoặc cần xử lý thêm.
        # Để đơn giản cho pipeline, ta sẽ xử lý sequence bên ngoài hoặc trong wrapper.
        self.model.eval()
        seq_length = self.config['models']['lstm'].get('n_lags', 30)
        
        # Tạo sequence cho tập test
        X_seq = []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:(i+seq_length)])
        X_seq = np.array(X_seq)
        
        inputs = torch.from_numpy(X_seq).float().to(DEVICE)
        with torch.no_grad():
            preds = self.model(inputs)
        
        # Pad lại phần đầu bằng NaN hoặc giá trị đầu tiên để khớp độ dài (nếu cần)
        # Ở đây trả về đúng kích thước dự báo
        return preds.cpu().numpy().flatten()