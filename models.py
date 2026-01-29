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
    def predict(self, *args, **kwargs):
        """Tạo dự báo. Các lớp con có thể nhận signature khác nhau
        (ví dụ `steps`, hoặc `test_df` + `steps`).
        """
        raise NotImplementedError
    
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
        """Lấy tên cột features thô (không bao gồm mục tiêu, dấu thời gian, và LAG features)"""
        # Loại trừ: các biến mục tiêu (intensity, y), cờ lọc (is_downtime), dấu thời gian, và LAG features
        # (LAG features sẽ được cập nhật từ predictions ở prediction time)
        exclude = ['intensity', 'y', 'requests', 'is_downtime', 'timestamp', 'ds']
        eng_cols = [col for col in df.columns if col not in exclude and col != target_col]
        # Lọc ra LAG features (lag_96, lag_672, lag_1440, lag_*, rolling_*)
        eng_cols = [col for col in eng_cols if not col.startswith('lag_') and not col.startswith('rolling_')]
        return eng_cols
    
    def _get_lag_feature_cols(self, df):
        """Lấy tên cột LAG features (lag_96, lag_672, lag_1440, etc)"""
        return [col for col in df.columns if col.startswith('lag_')]
    
    def _get_lag_features_from_sequence(self, sequence, lag_cols):
        """
        Tính lag features từ sequence (predictions) dựa trên lag_cols.
        
        Args:
            sequence: np.array giá trị (train + predicted)
            lag_cols: list tên lag features (e.g., ['lag_96', 'lag_672', 'lag_1440'])
        
        Returns:
            dict {lag_col: value} được tính từ sequence
        """
        lag_features = {}
        for lag_col in lag_cols:
            # Parse lag value từ column name (e.g., 'lag_96' → 96)
            try:
                lag_value = int(lag_col.split('_')[1])
                if len(sequence) > lag_value:
                    lag_features[lag_col] = sequence[-lag_value]
                else:
                    # Fallback nếu sequence chưa đủ dài
                    lag_features[lag_col] = sequence[0] if len(sequence) > 0 else 0.0
            except (ValueError, IndexError):
                lag_features[lag_col] = 0.0
        return lag_features

    def create_features(self, df, target_col):
       
        features = []
        targets = []
        
        # Lấy dữ liệu 
        data = df[target_col].values
        
        # Các cột features để bao gồm (cùng logic như trong fit()) - KHÔNG chứa lag
        feature_cols = self._get_eng_feature_cols(df, target_col)
        
        # Lấy tên lag columns để tính từ data array (không từ df)
        lag_cols = self._get_lag_feature_cols(df)

        for i in range(self.n_lags, len(data)):
            # Tạo các lag features từ data (rolling window)
            lags = data[i-self.n_lags:i]

            # Tính toán rolling statistics
            rolling_mean = np.mean(lags)
            rolling_std = np.std(lags)
            rolling_min = np.min(lags)
            rolling_max = np.max(lags)

            # Xu hướng đơn giản
            trend = lags[-1] - lags[0]

            # Tính LAG FEATURES từ data array (không từ df columns) ✅
            lag_features_dict = {}
            for lag_col in lag_cols:
                try:
                    lag_value = int(lag_col.split('_')[1])
                    if i >= lag_value:
                        lag_features_dict[lag_col] = data[i - lag_value]
                    else:
                        lag_features_dict[lag_col] = data[0]
                except (ValueError, IndexError):
                    lag_features_dict[lag_col] = 0.0
            
            lag_values = np.array([lag_features_dict.get(col, 0.0) for col in lag_cols]) if lag_cols else np.array([])

            # Lấy TRUE ENGINEERED FEATURES (không chứa lag) ✅
            eng_features = df.iloc[i][feature_cols].values if i < len(df) else np.zeros(len(feature_cols))

            # Kết hợp tất cả các features
            feature_vector = np.concatenate([
                lags,  
                [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                lag_values,  # ✅ LAG features từ data array
                eng_features  # ✅ TRUE engineered features
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
            self.lag_cols = self._get_lag_feature_cols(train_df)  # Lưu lag column names ✅
            self.feature_cols = self._get_eng_feature_cols(train_df, target_col)
            self.last_eng_features = train_df.iloc[-1][self.feature_cols].values.astype(np.float64)
            self.is_fitted = True

            logger.info(f"✓ XGBoost: {self.n_lags} lags + 5 rolling stats + {len(self.feature_cols)} features")
            return True

        except Exception as e:
            logger.error(f"Lỗi XGBoost: {e}")
            return False

    def predict(self, test_df=None, steps=24, target_col='intensity'):
        """Dự báo với XGBoost
        
        FIX Cách 2: Cập nhật lag features (kể cả lag_96, lag_672, lag_1440) từ PREDICTIONS 
        (chứ không phải từ test data thật) để đảm bảo consistency với behavior khi deploy.
        Engineered features (time-based, không phải lag) được giữ từ test_df.
        """
        if not self.is_fitted:
            logger.error("Mô hình chưa được huấn luyện")
            return None

        try:
            if test_df is not None and len(test_df) > 0:
                predictions = []
                sequence = self.last_sequence.copy()
                
                for idx in range(len(test_df)):
                    # Lấy LAG FEATURES từ SEQUENCE ĐỰ BÁO (updated từ predictions) ✅
                    lags = sequence[-self.n_lags:]
                    
                    # Tính ROLLING STATS từ SEQUENCE ĐỰ BÁO ✅
                    rolling_mean = np.mean(lags)
                    rolling_std = np.std(lags)
                    rolling_min = np.min(lags)
                    rolling_max = np.max(lags)
                    trend = lags[-1] - lags[0]
                    
                    # Cập nhật LAG FEATURES từ sequence (nếu có trong feature_cols) ✅
                    if hasattr(self, 'lag_cols') and self.lag_cols:
                        lag_feature_dict = self._get_lag_features_from_sequence(sequence, self.lag_cols)
                        # Xây dựng eng_features nhưng thay thế lag values từ predictions
                        eng_features = test_df.iloc[idx][self.feature_cols].values.astype(np.float64)
                        # Thêm lag features vào (từ sequence)
                        lag_values = np.array([lag_feature_dict.get(col, 0.0) for col in self.lag_cols])
                        feature_vector = np.concatenate([
                            lags,
                            [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                            lag_values,  # lag features từ predictions ✅
                            eng_features  # true engineered features từ test_df ✅
                        ]).reshape(1, -1)
                    else:
                        # Fallback nếu không có lag_cols
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
                    # Cập nhật SEQUENCE với PREDICTION để tính lag ở bước tiếp theo ✅
                    sequence = np.append(sequence[1:], pred)
                
                return np.array(predictions)
            else:
                predictions = []
                sequence = self.last_sequence.copy()

                for _ in range(steps):
                    # Cập nhật LAG + ROLLING STATS từ sequence (dự báo) ✅
                    rolling_mean = np.mean(sequence)
                    rolling_std = np.std(sequence)
                    rolling_min = np.min(sequence)
                    rolling_max = np.max(sequence)
                    trend = sequence[-1] - sequence[0]

                    eng_features = self.last_eng_features.copy() if hasattr(self, 'last_eng_features') else np.array([])
                    
                    # Cập nhật lag features từ sequence (nếu có) ✅
                    if hasattr(self, 'lag_cols') and self.lag_cols:
                        lag_feature_dict = self._get_lag_features_from_sequence(sequence, self.lag_cols)
                        lag_values = np.array([lag_feature_dict.get(col, 0.0) for col in self.lag_cols])
                        feature_vector = np.concatenate([
                            sequence,
                            [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                            lag_values,  # lag features từ predictions ✅
                            eng_features  # engineered features ✅
                        ]).reshape(1, -1)
                    else:
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
        """Lấy tên cột features thô (không bao gồm mục tiêu, dấu thời gian, và LAG features)"""
        exclude = ['intensity', 'y', 'requests', 'is_downtime', 'timestamp', 'ds']
        eng_cols = [col for col in df.columns if col not in exclude and col != target_col]
        # Lọc ra LAG features (lag_96, lag_672, lag_1440, lag_*, rolling_*)
        eng_cols = [col for col in eng_cols if not col.startswith('lag_') and not col.startswith('rolling_')]
        return eng_cols
    
    def _get_lag_feature_cols(self, df):
        """Lấy tên cột LAG features (lag_96, lag_672, lag_1440, etc)"""
        return [col for col in df.columns if col.startswith('lag_')]
    
    def _get_lag_features_from_sequence(self, sequence, lag_cols):
        """
        Tính lag features từ sequence (predictions) dựa trên lag_cols.
        
        Args:
            sequence: np.array giá trị unscaled (train + predicted)
            lag_cols: list tên lag features (e.g., ['lag_96', 'lag_672', 'lag_1440'])
        
        Returns:
            dict {lag_col: value} được tính từ sequence
        """
        lag_features = {}
        for lag_col in lag_cols:
            # Parse lag value từ column name (e.g., 'lag_96' → 96)
            try:
                lag_value = int(lag_col.split('_')[1])
                if len(sequence) > lag_value:
                    lag_features[lag_col] = sequence[-lag_value]
                else:
                    # Fallback nếu sequence chưa đủ dài
                    lag_features[lag_col] = sequence[0] if len(sequence) > 0 else 0.0
            except (ValueError, IndexError):
                lag_features[lag_col] = 0.0
        return lag_features

    def create_features(self, df, target_col):
        """Tạo feature matrix multivariate cho LSTM"""
        features = []
        targets = []
        
        data = df[target_col].values
        feature_cols = self._get_eng_feature_cols(df, target_col)
        lag_cols = self._get_lag_feature_cols(df)

        for i in range(self.n_lags, len(data)):
            lags = data[i-self.n_lags:i]
            # Rolling stats từ lags original (sẽ được scale cùng với lags)
            rolling_mean = np.mean(lags)
            rolling_std = np.std(lags)
            rolling_min = np.min(lags)
            rolling_max = np.max(lags)
            trend = lags[-1] - lags[0]
            
            # Tính LAG FEATURES từ data array (không từ df columns) ✅
            lag_features_dict = {}
            for lag_col in lag_cols:
                try:
                    lag_value = int(lag_col.split('_')[1])
                    if i >= lag_value:
                        lag_features_dict[lag_col] = data[i - lag_value]
                    else:
                        lag_features_dict[lag_col] = data[0]
                except (ValueError, IndexError):
                    lag_features_dict[lag_col] = 0.0
            
            lag_values = np.array([lag_features_dict.get(col, 0.0) for col in lag_cols]) if lag_cols else np.array([])
            
            # Lấy TRUE ENGINEERED FEATURES (không chứa lag) ✅
            eng_features = df.iloc[i][feature_cols].values if i < len(df) else np.zeros(len(feature_cols))
            
            feature_vector = np.concatenate([
                lags,
                [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                lag_values,  # ✅ LAG features từ data array
                eng_features  # ✅ TRUE engineered features
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
            self.lag_cols = self._get_lag_feature_cols(train_df)  # Lưu lag column names ✅
            self.feature_cols = self._get_eng_feature_cols(train_df, target_col)
            self.is_fitted = True
            logger.info(f"✓ LSTM: {self.n_lags} lags + 5 rolling stats + {len(self.lag_cols)} lag features + {len(self.feature_cols)} other features")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi LSTM: {e}")
            return False
    
    def predict(self, test_df=None, steps=24, target_col='intensity'):
        """Dự báo với LSTM multivariate
        
        FIX Cách 2: Khi test_df được cung cấp, cập nhật lag features (kể cả lag_96, lag_672, etc)
        và rolling stats từ PREDICTIONS (chứ không phải từ test data thật) để đảm bảo consistency.
        Engineered features (time-based, không phải lag) được giữ từ test_df.
        """
        if not self.is_fitted:
            logger.error("Mô hình chưa được huấn luyện")
            return None
        
        try:
            if test_df is not None and len(test_df) > 0:
                predictions = []
                
                # Khởi tạo sequence từ train data (context) - scaled values
                sequence = self.last_sequence.flatten().copy()  # shape (n_features,) - scaled
                feature_cols = self._get_eng_feature_cols(test_df, target_col)
                lag_cols = self._get_lag_feature_cols(test_df)
                
                # Để cập nhật lag features từ predictions, ta cần track unscaled values
                sequence_unscaled_target = np.array([])  # Will accumulate target predictions
                
                for idx in range(len(test_df)):
                    # Lấy LAG FEATURES từ SEQUENCE ĐỰ BÁO (không phải test data thật) ✅
                    lags = sequence[:self.n_lags]
                    
                    # Tính ROLLING STATS từ LAG ĐỰ BÁO ✅
                    rolling_mean = np.mean(lags)
                    rolling_std = np.std(lags)
                    rolling_min = np.min(lags)
                    rolling_max = np.max(lags)
                    trend = lags[-1] - lags[0]
                    
                    # Giữ TRUE ENGINEERED FEATURES từ test_df (time-based, không phải lag) ✅
                    eng_features_raw = test_df.iloc[idx][feature_cols].values if len(feature_cols) > 0 else np.array([])
                    
                    # Xây dựng feature vector với lag + rolling stats từ sequence (predictions)
                    feature_vector = np.concatenate([
                        lags,
                        [rolling_mean, rolling_std, rolling_min, rolling_max, trend],
                        eng_features_raw
                    ])
                    
                    # Scale và reshape cho LSTM
                    feature_vector_scaled = self.feature_scaler.transform([feature_vector])[0]
                    X_lstm = feature_vector_scaled.reshape(1, len(feature_vector_scaled), 1)
                    
                    # Dự báo
                    pred_scaled = self.model.predict(X_lstm, verbose=0)[0, 0]
                    pred_unscaled = self.target_scaler.inverse_transform([[pred_scaled]])[0, 0]
                    pred_unscaled = max(pred_unscaled, 0)
                    
                    predictions.append(pred_unscaled)
                    
                    # Cập nhật SEQUENCE với PREDICTION để tính lag ở bước tiếp theo ✅
                    # Cập nhật scaled sequence (shift left, add pred_scaled)
                    sequence = np.concatenate([sequence[1:], [pred_scaled]])
                    
                    # Track unscaled target values để có thể cập nhật lag features chính xác
                    sequence_unscaled_target = np.append(sequence_unscaled_target, pred_unscaled)
                
                return np.array(predictions)
            else:
                predictions = []
                sequence = self.last_sequence.flatten().copy()  # shape (n_features,) - scaled
                
                for _ in range(steps):
                    X_pred = sequence.reshape(1, len(sequence), 1)
                    pred_scaled = self.model(X_pred, training=False)[0, 0].numpy()
                    
                    pred_unscaled = self.target_scaler.inverse_transform([[pred_scaled]])[0, 0]
                    pred_unscaled = max(pred_unscaled, 0)
                    predictions.append(pred_unscaled)
                    
                    # Cập nhật chuỗi với dự báo mới (LAG + ROLLING STATS từ predictions) ✅
                    lags = sequence[:self.n_lags]
                    lags_updated = np.concatenate([lags[1:], [pred_scaled]])
                    
                    # Tính rolling stats mới từ predictions
                    rolling_mean = np.mean(lags_updated)
                    rolling_std = np.std(lags_updated)
                    rolling_min = np.min(lags_updated)
                    rolling_max = np.max(lags_updated)
                    trend = lags_updated[-1] - lags_updated[0]
                    
                    # Lấy engineered features (không đổi khi không có test_df)
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
        if model_name in ['prophet']:
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