"""
MODULE: EVALUATION PIPELINE
"""
import pandas as pd
import numpy as np
import joblib
import torch
import yaml
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import ProphetForecaster, XGBoostForecaster, LSTMForecaster # Import class wrapper

# Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FinalFixerNoLog:
    def __init__(self):
        self.models_dir = Path("models")
        self.data_dir = Path("data")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.target_col = 'intensity'
        
        try:
            with open("config/config.yaml", "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            self.intervals = self.config['processing'].get('intervals', ['1min', '5min', '15min'])
        except:
            self.intervals = ['1min', '5min', '15min']
        
        print(f"📋 Danh sách chạy: {self.intervals}")

        self.feature_cols = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend',
            'lag_1step', 'lag_1h', 'lag_24h', 'lag_7d',
            'roll_mean_4h', 'roll_std_4h', 'roll_max_4h'
        ]

    def calculate_metrics(self, y_true, y_pred, interval):
        # 1. Chặn số âm 
        y_pred = np.maximum(y_pred, 0)
        
        # 2. Cắt độ dài
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]

        # 3. WARM-UP CUT (Bỏ 24h đầu để model ổn định)
        if '1min' in interval: cut = 1440
        elif '5min' in interval: cut = 288
        elif '15min' in interval: cut = 96
        else: cut = 0
        
        if len(y_true) > cut:
            y_true_s = y_true[cut:]
            y_pred_s = y_pred[cut:]
        else:
            y_true_s = y_true
            y_pred_s = y_pred
        
        # 4. Tính toán Metrics
        mse = mean_squared_error(y_true_s, y_pred_s)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_s, y_pred_s)
        
        # 5. MAPE (Lọc bỏ các điểm thực tế = 0 để tránh chia cho 0)
        mask = y_true_s > 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true_s[mask] - y_pred_s[mask]) / y_true_s[mask])) * 100
        else:
            mape = 0.0 
            
        return rmse, mse, mae, mape

    def run(self):
        leaderboard = []
        print(f"\n{'='*60}")
        print(f"🚀 CHẠY FINAL FIXER (NO LOG - MATCHING TRAIN.PY)")
        print(f"{'='*60}")

        for interval in self.intervals:
            print(f"\n📂 Interval: {interval}")
            train_path = self.data_dir / f"prepared_train_{interval}.csv"
            test_path = self.data_dir / f"prepared_test_{interval}.csv"
            
            if not train_path.exists(): continue
                
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            df_test['ds'] = pd.to_datetime(df_test['ds'])
            y_true = df_test[self.target_col].values 
            df_test = df_test.reset_index(drop=True)
            
            df_preds = pd.DataFrame({'ds': df_test['ds'], 'Actual': y_true})

            # 1. PROPHET
            try:
                prophet_path = self.models_dir / f"prophet_{interval}.pkl"
                if prophet_path.exists():
                    # Load model đã train từ file .pkl
                    model_p = joblib.load(prophet_path)
                    
                    # Dự báo (Hàm predict của ProphetForecaster đã xử lý việc gọi model.predict)
                    # Lưu ý: df_test cần có đủ các cột regressor mà model đã học
                    pred_p = model_p.predict(df_test)[-len(y_true):]
                
                    rmse, mse, mae, mape = self.calculate_metrics(y_true, pred_p, interval)
                    leaderboard.append({'Interval': interval, 'Model': 'Prophet', 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE (%)': mape})
                    df_preds['Prophet'] = pred_p
                    print(f"   ✅ Prophet: MAPE={mape:.2f}%")
            except Exception as e: print(f"   ❌ Prophet Error: {e}")

            # 2. XGBOOST
            try:
                xgb_path = self.models_dir / f"xgboost_{interval}.pkl"
                if xgb_path.exists():
                    model_xgb = joblib.load(xgb_path)
                    scaler_X = joblib.load(self.models_dir / f"scaler_X_{interval}.pkl")
                    scaler_y = joblib.load(self.models_dir / f"scaler_y_{interval}.pkl")
                    
                    feat_cols = [c for c in self.feature_cols if c in df_test.columns]
                    X_scaled = scaler_X.transform(df_test[feat_cols].values)
                    
                    # Predict -> Inverse Scaler
                    pred_xgb = scaler_y.inverse_transform(model_xgb.predict(X_scaled).reshape(-1, 1)).flatten()
                    
                    rmse, mse, mae, mape = self.calculate_metrics(y_true, pred_xgb, interval)
                    leaderboard.append({'Interval': interval, 'Model': 'XGBoost', 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE (%)': mape})
                    df_preds['XGBoost'] = pred_xgb
                    print(f"   ✅ XGBoost: MAPE={mape:.2f}%")
            except Exception as e: print(f"   ⚠️ XGBoost Error: {e}")

            # 3. LSTM
            try:
                lstm_path = self.models_dir / f"lstm_{interval}.pth"
                if lstm_path.exists():
                    input_dim = len([c for c in self.feature_cols if c in df_test.columns])
                    # Sử dụng config gốc để đảm bảo tham số khớp với lúc train
                    forecaster = LSTMForecaster(self.config, input_dim)
                    forecaster.model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
                    forecaster.model.eval()
                    
                    scaler_X = joblib.load(self.models_dir / f"scaler_X_{interval}.pkl")
                    scaler_y = joblib.load(self.models_dir / f"scaler_y_{interval}.pkl")
                    
                    feat_cols = [c for c in self.feature_cols if c in df_test.columns]
                    X_scaled = scaler_X.transform(df_test[feat_cols].values)
                    
                    n_lags = self.config['models']['lstm'].get('n_lags', 30)
                    if len(X_scaled) > n_lags:
                        X_seq = np.array([X_scaled[i:i+n_lags] for i in range(len(X_scaled)-n_lags)])
                        inp = torch.from_numpy(X_seq).float().to(DEVICE)
                        with torch.no_grad():
                            p = forecaster.model(inp).cpu().numpy().flatten()
                        
                        # Inverse Scaler 
                        pred_lstm = scaler_y.inverse_transform(p.reshape(-1, 1)).flatten()
                        
                        y_trim = y_true[n_lags:]
                        rmse, mse, mae, mape = self.calculate_metrics(y_trim, pred_lstm, interval)
                        leaderboard.append({'Interval': interval, 'Model': 'LSTM', 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE (%)': mape})
                        df_preds['LSTM'] = np.concatenate([[np.nan]*n_lags, pred_lstm])
                        print(f"   ✅ LSTM: MAPE={mape:.2f}%")
            except Exception as e: print(f"   ⚠️ LSTM Error: {e}")

            # Lưu file CSV cho visualize.py
            df_preds.to_csv(self.results_dir / f"predictions_{interval}.csv", index=False)

        if leaderboard:
            df = pd.DataFrame(leaderboard).sort_values(by=['Interval', 'RMSE'])
            df = df[['Interval', 'Model', 'RMSE', 'MSE', 'MAE', 'MAPE (%)']]
            print(f"\n{'='*60}\n🏆 FINAL LEADERBOARD (NO LOG)\n{'='*60}")
            print(df.to_string(index=False))
            df.to_csv(self.results_dir / "final_leaderboard.csv", index=False)

if __name__ == "__main__":
    FinalFixerNoLog().run()