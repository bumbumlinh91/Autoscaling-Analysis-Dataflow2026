"""
Mô hình dự báo chuỗi thời gian cho Dataflow 2026
Triển khai: Prophet, XGBoost, LSTM
"""
import numpy as np
import pandas as pd
import warnings
from abc import ABC, abstractmethod
import logging
import pickle
import yaml
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ================================================================
# 1. IMPORT CÁC THƯ VIỆN MÔ HÌNH
# ================================================================

try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        logger.warning("Prophet không khả dụng")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError:
    logger.warning("sklearn không khả dụng")

try:
    import xgboost as xgb
except ImportError:
    logger.warning("XGBoost không khả dụng")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    logger.warning("TensorFlow/Keras không khả dụng")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    logger.warning("statsmodels không khả dụng")


# ================================================================
# 2. LỚP CƠ SỞ CHO TẤT CẢ CÁC DỰ BÁO
# ================================================================

class BaseForecaster(ABC):
    """Lớp cơ sở cho tất cả các dự báo"""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, train_data):
        """Huấn luyện mô hình trên dữ liệu huấn luyện"""
        pass
    
    @abstractmethod
    def predict(self, steps):
        """Tạo dự báo cho 'steps' kỳ tiếp theo"""
        pass
    
    def evaluate(self, y_true, y_pred):
        """Tính toán các chỉ số đánh giá: RMSE, MSE, MAE, MAPE"""
        # Đảm bảo các đầu vào là mảng numpy
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        
        # Xóa các giá trị NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'rmse': 0, 'mse': 0, 'mae': 0, 'mape': 0}
        
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # MAPE tính toán chỉ trên các giá trị thực tế khác 0 để tránh chia cho 0
        mask_nonzero = y_true_clean != 0
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((y_true_clean[mask_nonzero] - y_pred_clean[mask_nonzero]) / y_true_clean[mask_nonzero])) * 100
        else:
            mape = 0.0
        
        return {
            'rmse': float(rmse),
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape)
        }
    
    def save(self, filepath):
        """Lưu mô hình vào tệp"""
        raise NotImplementedError
    
    def load(self, filepath):
        """Tải mô hình từ tệp"""
        raise NotImplementedError


# ================================================================
# 3. PROPHET DỰ BÁO
# ================================================================

class ProphetForecaster(BaseForecaster):
    """Facebook Prophet để dự báo chuỗi thời gian"""
    
    def __init__(self, daily_seasonality=True, weekly_seasonality=True, 
                 yearly_seasonality=False, changepoint_prior_scale=0.05, 
                 interval_width=0.95, freq='D', **kwargs):
        super().__init__("Prophet")
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width
        self.freq = freq
        self.last_timestamp = None
    
    def fit(self, train_data):
        """
        Huấn luyện mô hình Prophet
        Args:
            train_data: pandas Series có chỉ mục datetime
        """
        try:
            # Chuẩn bị dữ liệu cho Prophet (yêu cầu các cột 'ds' và 'y')
            df = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data.values
            })
            
            # Áp dụng log-transform để xử lý phân phối lệch phải (Độ lệch = 1,41 theo EDA)
            df['y'] = np.log1p(df['y'])
            
            # Khởi tạo Prophet
            self.model = Prophet(
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                interval_width=self.interval_width
            )
            
            # Huấn luyện mô hình
            self.model.fit(df)
            self.last_timestamp = train_data.index[-1]
            self.is_fitted = True
            
            logger.info(f"✓ Mô hình Prophet đã được huấn luyện thành công (với log-transform để xử lý độ lệch)")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện Prophet: {e}")
            return False
    
    def predict(self, steps=24):
        """Dự báo 'steps' kỳ tiếp theo"""
        if not self.is_fitted:
            logger.error("Mô hình chưa được huấn luyện")
            return None
        
        try:
            # Tạo DataFrame trong tương lai
            future = self.model.make_future_dataframe(periods=steps, freq=self.freq)
            
            # Tạo dự báo
            forecast = self.model.predict(future)
            
            # Trả về chỉ các dự báo trong tương lai
            predictions = forecast['yhat'].iloc[-steps:].values
            
            # Đảo ngược log-transform để quay lại tỷ lệ ban đầu
            predictions = np.expm1(predictions)
            
            # Đảm bảo dự báo không âm
            predictions = np.maximum(predictions, 0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Lỗi khi dự báo với Prophet: {e}")
            return None
    
    def save(self, filepath):
        """Lưu mô hình Prophet"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'last_timestamp': self.last_timestamp,
                'params': {
                    'daily_seasonality': self.daily_seasonality,
                    'weekly_seasonality': self.weekly_seasonality,
                    'yearly_seasonality': self.yearly_seasonality,
                    'changepoint_prior_scale': self.changepoint_prior_scale,
                    'interval_width': self.interval_width,
                    'freq': self.freq
                }
            }, f)
    
    def load(self, filepath):
        """Tải mô hình Prophet"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.last_timestamp = data['last_timestamp']
            params = data['params']
            self.daily_seasonality = params['daily_seasonality']
            self.weekly_seasonality = params['weekly_seasonality']
            self.yearly_seasonality = params['yearly_seasonality']
            self.changepoint_prior_scale = params['changepoint_prior_scale']
            self.interval_width = params.get('interval_width', 0.95)
            self.freq = params.get('freq', 'D')
            self.is_fitted = True


# ================================================================
# 4. XGBOOST DỰ BÁO
# ================================================================

class XGBoostForecaster(BaseForecaster):
    """XGBoost để hồi quy chuỗi thời gian với kỹ thuật làm giàu tự động"""
    
    def __init__(self, n_lags=60, n_estimators=300, max_depth=7, 
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 objective='reg:squarederror', random_state=42, **kwargs):
        super().__init__("XGBoost")
        self.n_lags = n_lags
        self.params = {
            'objective': objective,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state
        }
        self.last_sequence = None

    def _get_eng_feature_cols(self, df, target_col):
        """Lấy tên cột features thô (không bao gồm mục tiêu và dấu thời gian)"""
        # Loại trừ: các biến mục tiêu (intensity, y), cờ lọc (is_downtime), và dấu thời gian
        exclude = ['intensity', 'y', 'requests', 'is_downtime', 'timestamp', 'ds']
        return [col for col in df.columns if col not in exclude and col != target_col]

    def create_features(self, df, target_col):
       
        features = []
        targets = []
        
        # Lấy dữ liệu 
        data = df[target_col].values
        
        # Các cột features để bao gồm (cùng logic như trong fit())
        feature_cols = self._get_eng_feature_cols(df, target_col)

        for i in range(self.n_lags, len(data)):
            # Tạo các lag features
            lags = data[i-self.n_lags:i]

            # Tính toán rolling statistics
            rolling_mean = np.mean(lags)
            rolling_std = np.std(lags)
            rolling_min = np.min(lags)
            rolling_max = np.max(lags)

            # Xu hướng đơn giản
            trend = lags[-1] - lags[0]

            # Lấy các features tại thời điểm i
            eng_features = df.iloc[i][feature_cols].values if i < len(df) else np.zeros(len(feature_cols))

            # Kết hợp tất cả các features
            feature_vector = np.concatenate([
                lags,  
                [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                eng_features
            ])

            features.append(feature_vector)
            targets.append(data[i])

        return np.array(features), np.array(targets)

    def fit(self, train_df, target_col='requests'):
        """Huấn luyện mô hình XGBoost (không scale)"""
        try:
            X, y = self.create_features(train_df, target_col)

            if len(X) == 0:
                logger.error("Không đủ dữ liệu")
                return False

            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X, y, verbose=False)
            self.last_sequence = train_df[target_col].values[-self.n_lags:].copy()
            self.feature_cols = self._get_eng_feature_cols(train_df, target_col)
            self.last_eng_features = train_df.iloc[-1][self.feature_cols].values.astype(np.float64)
            self.is_fitted = True

            logger.info(f"✓ XGBoost: {self.n_lags} lags + 5 rolling stats + {len(self.feature_cols)} features")
            return True

        except Exception as e:
            logger.error(f"Lỗi XGBoost: {e}")
            return False

    def predict(self, test_df=None, steps=24, target_col='intensity'):
        """Dự báo với XGBoost"""
        if not self.is_fitted:
            logger.error("Mô hình chưa được huấn luyện")
            return None

        try:
            if test_df is not None and len(test_df) > 0:
                predictions = []
                sequence = self.last_sequence.copy()
                
                for idx in range(len(test_df)):
                    lags = sequence[-self.n_lags:]
                    rolling_mean = np.mean(lags)
                    rolling_std = np.std(lags)
                    rolling_min = np.min(lags)
                    rolling_max = np.max(lags)
                    trend = lags[-1] - lags[0]
                    
                    eng_features = test_df.iloc[idx][self.feature_cols].values.astype(np.float64)
                    
                    feature_vector = np.concatenate([
                        lags,
                        [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                        eng_features
                    ]).reshape(1, -1)
                    
                    # XGBoost không scale - dự báo trực tiếp
                    pred = self.model.predict(feature_vector)[0]
                    pred = max(pred, 0)
                    
                    predictions.append(pred)
                    sequence = np.append(sequence[1:], pred)
                
                return np.array(predictions)
            else:
                predictions = []
                sequence = self.last_sequence.copy()

                for _ in range(steps):
                    rolling_mean = np.mean(sequence)
                    rolling_std = np.std(sequence)
                    rolling_min = np.min(sequence)
                    rolling_max = np.max(sequence)
                    trend = sequence[-1] - sequence[0]

                    eng_features = self.last_eng_features.copy() if hasattr(self, 'last_eng_features') else np.array([])

                    feature_vector = np.concatenate([
                        sequence,
                        [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                        eng_features
                    ]).reshape(1, -1)

                    # XGBoost không scale - dự báo trực tiếp
                    pred = self.model.predict(feature_vector)[0]
                    pred = max(pred, 0)
                    predictions.append(pred)
                    sequence = np.append(sequence[1:], pred)

                return np.array(predictions)

        except Exception as e:
            logger.error(f"Lỗi khi dự báo với XGBoost: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save(self, filepath):
        """Lưu mô hình XGBoost"""
        with open(filepath + '.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'last_sequence': self.last_sequence,
                'n_lags': self.n_lags,
                'last_eng_features': self.last_eng_features,
                'feature_cols': self.feature_cols
            }, f)
    
    def load(self, filepath):
        """Tải mô hình XGBoost"""
        with open(filepath + '.pkl', 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.last_sequence = data['last_sequence']
            self.n_lags = data['n_lags']
            self.last_eng_features = data.get('last_eng_features', np.array([]))
            self.feature_cols = data.get('feature_cols', [])
            self.is_fitted = True


# ================================================================
# 5. LSTM DỰ BÁO
# ================================================================

class LSTMForecaster(BaseForecaster):
    """Mô hình Học Sâu LSTM  cho chuỗi thời gian """
    
    def __init__(self, n_lags=60, units=128, epochs=100, batch_size=32, 
                 dropout=0.2, learning_rate=0.001, optimizer='adam', loss='mse', 
                 early_stopping=None, **kwargs):
        super().__init__("LSTM")
        self.n_lags = n_lags
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.loss = loss
        
        if isinstance(early_stopping, dict):
            self.early_stopping_enabled = early_stopping.get('enabled', True)
            self.early_stopping_patience = early_stopping.get('patience', 15)
        else:
            self.early_stopping_enabled = True
            self.early_stopping_patience = 15

        self.target_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.last_sequence = None
        self.feature_cols = []

    def _get_eng_feature_cols(self, df, target_col):
        """Lấy tên cột features thô"""
        exclude = ['intensity', 'y', 'requests', 'is_downtime', 'timestamp', 'ds']
        return [col for col in df.columns if col not in exclude and col != target_col]

    def create_features(self, df, target_col):
        """Tạo feature matrix multivariate cho LSTM"""
        features = []
        targets = []
        
        data = df[target_col].values
        feature_cols = self._get_eng_feature_cols(df, target_col)

        for i in range(self.n_lags, len(data)):
            lags = data[i-self.n_lags:i]
            # Rolling stats từ lags original (sẽ được scale cùng với lags)
            rolling_mean = np.mean(lags)
            rolling_std = np.std(lags)
            rolling_min = np.min(lags)
            rolling_max = np.max(lags)
            trend = lags[-1] - lags[0]
            
            eng_features = df.iloc[i][feature_cols].values if i < len(df) else np.zeros(len(feature_cols))
            
            feature_vector = np.concatenate([
                lags,
                [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                eng_features
            ])
            
            features.append(feature_vector)
            targets.append(data[i])

        return np.array(features), np.array(targets)

    def fit(self, train_df, target_col='requests'):
        """Huấn luyện mô hình LSTM"""
        try:
            X, y = self.create_features(train_df, target_col)

            if len(X) == 0:
                logger.error("Không đủ dữ liệu để tạo các tính năng")
                return False

            # Scale features và targets cho LSTM
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            
            samples, n_features = X_scaled.shape
            X_lstm = X_scaled.reshape((samples, n_features, 1))

            self.model = Sequential([
                LSTM(self.units, activation='relu', return_sequences=True, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
                Dropout(self.dropout),
                LSTM(self.units // 2, activation='relu', return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.units // 4, activation='relu'),
                Dropout(self.dropout),
                Dense(16, activation='relu'),
                Dense(1)
            ])

            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss=self.loss,
                metrics=['mae']
            )

            early_stop = EarlyStopping(
                monitor='loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            )

            callbacks = [early_stop] if self.early_stopping_enabled else []

            self.model.fit(
                X_lstm, y_scaled,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=0
            )

            self.last_sequence = X_scaled[-1:].copy()
            self.feature_cols = self._get_eng_feature_cols(train_df, target_col)
            self.is_fitted = True
            logger.info(f"✓ LSTM: {self.n_lags} lags + 5 rolling stats + {len(self.feature_cols)} features")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi LSTM: {e}")
            return False
    
    def predict(self, test_df=None, steps=24, target_col='intensity'):
        """Dự báo với LSTM multivariate"""
        if not self.is_fitted:
            logger.error("Mô hình chưa được huấn luyện")
            return None
        
        try:
            if test_df is not None and len(test_df) > 0:
                X_test, _ = self.create_features(test_df, target_col)
                X_test_scaled = self.feature_scaler.transform(X_test)
                
                # Reshape để LSTM: (samples, n_features, 1)
                X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
                
                pred_scaled = self.model.predict(X_test_lstm, verbose=0).flatten()
                predictions = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                predictions = np.maximum(predictions, 0)
                
                return predictions
            else:
                
                predictions = []
                sequence = self.last_sequence.flatten().copy()  # shape (n_features,)
                
                for _ in range(steps):
                    X_pred = sequence.reshape(1, len(sequence), 1)
                    pred_scaled = self.model(X_pred, training=False)[0, 0].numpy()
                    
                    pred_unscaled = self.target_scaler.inverse_transform([[pred_scaled]])[0, 0]
                    pred_unscaled = max(pred_unscaled, 0)
                    predictions.append(pred_unscaled)
                    
                    # Cập nhật chuỗi với dự báo mới
                    lags = sequence[:self.n_lags]
                    lags_updated = np.concatenate([lags[1:], [pred_scaled]])
                    
                    # tính toán rolling stats mới
                    rolling_mean = np.mean(lags_updated)
                    rolling_std = np.std(lags_updated)
                    rolling_min = np.min(lags_updated)
                    rolling_max = np.max(lags_updated)
                    trend = lags_updated[-1] - lags_updated[0]
                    
                    # Lấy các features thô cuối cùng 
                    n_eng_features = len(sequence) - self.n_lags - 5
                    eng_features = sequence[-n_eng_features:] if n_eng_features > 0 else np.array([])
                    
                    sequence = np.concatenate([
                        lags_updated,
                        [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                        eng_features
                    ])
                
                return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Lỗi khi dự báo với LSTM: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save(self, filepath):
        """Lưu mô hình LSTM dưới dạng pickle"""
        with open(filepath + '.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'target_scaler': self.target_scaler,
                'last_sequence': self.last_sequence,
                'n_lags': self.n_lags,
                'units': self.units
            }, f)
    
    def load(self, filepath):
        """Tải mô hình LSTM từ pickle"""
        with open(filepath + '.pkl', 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.target_scaler = data.get('target_scaler', StandardScaler())
            self.last_sequence = data['last_sequence']
            self.n_lags = data['n_lags']
            self.units = data['units']
            self.is_fitted = True


# ================================================================
# 6. HÀM TIỆN ÍCH
# ================================================================

def create_forecaster(model_name, config_path=None, **kwargs):
    """
    Hàm nhà máy để tạo các dự báo
    Args:
        model_name: Tên của mô hình ('prophet', 'xgboost', 'lstm', 'sarima')
        config_path: Đường dẫn đến tệp config.yaml (tùy chọn, sử dụng mặc định nếu không được cung cấp)
        **kwargs: Các tham số bổ sung để ghi đè các giá trị config
    """
    model_map = {
        'prophet': ProphetForecaster,
        'xgboost': XGBoostForecaster,
        'lstm': LSTMForecaster
    }
    
    model_name_lower = model_name.lower()
    if model_name_lower not in model_map:
        raise ValueError(f"Mô hình không xác định: {model_name}. Có sẵn: {list(model_map.keys())}")
    
    # Tải config nếu được cung cấp
    model_config = {}
    if config_path:
        config = load_config(config_path)
        if model_name_lower in config.get('models', {}):
            model_cfg = config['models'][model_name_lower]
            # Trích xuất các tham số từ phần 'params'
            model_config.update(model_cfg.get('params', {}))
            # Cũng trích xuất các khóa cấp cao nhất như 'n_lags' (không bao gồm 'enabled')
            for key, value in model_cfg.items():
                if key not in ('params', 'enabled'):
                    model_config[key] = value
    
    # Hợp nhất với kwargs (kwargs có ưu tiên)
    final_params = {**model_config, **kwargs}
    
    return model_map[model_name_lower](**final_params)


def load_config(config_path):
    """Tải cấu hình từ tệp YAML"""
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Tệp cấu hình không tìm thấy: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Cấu hình đã tải từ {config_path}")
        return config if config else {}
    
    except Exception as e:
        logger.error(f"Lỗi khi tải cấu hình: {e}")
        return {}


# ================================================================
# 7. KIỂM TRA VÀ TEST
# ================================================================

if __name__ == "__main__":
    # Mã kiểm tra
    logging.basicConfig(level=logging.INFO)
    
    # Tải cấu hình
    # models.py nằm trong gốc dự án, config/ nằm một cấp dưới
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    config = load_config(config_path)
    
    # Tạo dữ liệu kiểm tra
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1000, freq='min')
    values = 100 + 20 * np.sin(2 * np.pi * np.arange(1000) / 60) + np.random.normal(0, 5, 1000)
    # Tạo dữ liệu kiểm tra
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=1000, freq='min')
    values = 100 + 20 * np.sin(2 * np.pi * np.arange(1000) / 60) + np.random.normal(0, 5, 1000)
    
    # Tạo DataFrame với các featurescho XGBoost/LSTM
    error_4xx = 5 + np.random.normal(0, 2, 1000)
    error_5xx = 2 + np.random.normal(0, 1, 1000)
    train_df = pd.DataFrame({
        'requests': values,
        'avg_bytes': 100 + np.random.normal(0, 10, 1000),
        'error_4xx': error_4xx,
        'error_5xx': error_5xx,
        'error_rate': (error_4xx + error_5xx) / (values + 1e-8),
        'intensity': values * (100 + np.random.normal(0, 10, 1000)),
        'is_downtime': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
        'hour': dates.hour,
        'day_of_week': dates.dayofweek,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    }, index=dates)
    
    # Cũng giữ Series cho Prophet
    train_data = pd.Series(values, index=dates)
    
    print("\nĐang kiểm tra các dự báo...")
    
    # Lấy các mô hình được bật từ config
    models_config = config.get('models', {})
    for model_name in ['prophet', 'xgboost', 'lstm']:
        model_cfg = models_config.get(model_name, {})
        
        # Bỏ qua nếu không được bật
        if not model_cfg.get('enabled', False):
            print(f"\n⊘ {model_name.upper()} bị vô hiệu hóa trong config")
            continue
        
        print(f"\n{'='*50}")
        print(f"Đang kiểm tra {model_name.upper()}")
        print('='*50)
        
        # Tạo dự báo bằng config
        model = create_forecaster(model_name, config_path=config_path)
        
        # Sử dụng dữ liệu thích hợp dựa trên loại mô hình
        if model_name in ['prophet', 'sarima']:
            fit_result = model.fit(train_data)
        else:
            fit_result = model.fit(train_df, target_col='requests')
        
        if fit_result:
            predictions = model.predict(steps=24)
            if predictions is not None:
                print(f"Dự báo (5 quan sát đầu tiên): {predictions[:5]}")
            else:
                print(f"⚠ {model_name.upper()} predict() trả về None")
        else:
            print(f"⚠ {model_name.upper()} fit() thất bại")