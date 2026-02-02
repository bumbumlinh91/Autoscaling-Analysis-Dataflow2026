"""
SCRIPT: EVALUATION RUNNER
--------------------------------------------
Mô tả:
    Kịch bản đánh giá hiệu năng mô hình.
    Tự động load dữ liệu test, model, scaler.
    Tính toán các chỉ số đánh giá và xuất bảng xếp hạng.
"""
import sys
import yaml
import logging
import joblib
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- [FIX 1: SETUP ĐƯỜNG DẪN GỐC] ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- [FIX 2: IMPORT TỪ SRC] ---
from src.models import ProphetForecaster, XGBoostForecaster, LSTMForecaster

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [Optional] Hack module
sys.modules['models'] = sys.modules['src.models']

class Evaluator:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        
        # Đường dẫn (Lưu ý: Train lưu ở đâu thì Evaluate phải đọc ở đó)
        # Nếu train lưu ở 'models', hãy sửa dòng dưới thành 'models'
        self.models_dir = self.project_root / "saved_models"
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.target_col = 'intensity'
        
        # Load Config
        config_path = self.project_root / "config/config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            self.intervals = self.config['processing'].get('intervals', ['15min'])
        except Exception as e:
            print(f"⚠️ Dùng default intervals do lỗi config: {e}")
            self.intervals = ['15min']
        
        print(f"📋 Danh sách đánh giá: {self.intervals}")
        print(f"📂 Thư mục Model: {self.models_dir}")

    def calculate_metrics(self, y_true, y_pred, interval):
        y_pred = np.maximum(y_pred, 0)
        
        # Cắt độ dài
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]

        # WARM-UP CUT
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
        
        mse = mean_squared_error(y_true_s, y_pred_s)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_s, y_pred_s)
        
        mask = y_true_s > 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_true_s[mask] - y_pred_s[mask]) / y_true_s[mask])) * 100
        else:
            mape = 0.0 
            
        return rmse, mse, mae, mape

    def load_test_data(self, interval):
        path = self.data_dir / f"prepared_test_{interval}.csv"
        if not path.exists():
             path = self.data_dir / f"processed_test_{interval}.csv"
        
        if not path.exists():
            return None
        
        df = pd.read_csv(path)
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.sort_values('ds').reset_index(drop=True)
        return df

    def run(self):
        leaderboard = []
        print(f"\n{'='*60}")
        print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ (AUTO FIX FEATURE MISMATCH)")
        print(f"{'='*60}")

        for interval in self.intervals:
            print(f"\n📂 Interval: {interval}")
            
            # 1. Load Data
            df_test = self.load_test_data(interval)
            if df_test is None: 
                print(f"   ❌ Không tìm thấy data test tại {self.data_dir}")
                continue
            
            # 2. Dynamic Feature Selection (Lấy tất cả feature tiềm năng)
            exclude_cols = ['ds', 'timestamp', self.target_col, 'y']
            feature_cols = [c for c in df_test.columns if c not in exclude_cols]
            
            y_true = df_test[self.target_col].values 
            df_preds = pd.DataFrame({'ds': df_test['ds'], 'Actual': y_true})

            # --- MODEL 1: PROPHET ---
            try:
                model_path = self.models_dir / f"prophet_{interval}.pkl"
                if model_path.exists():
                    model_p = joblib.load(model_path)
                    pred_p = model_p.predict(df_test)[-len(y_true):]
                    
                    rmse, mse, mae, mape = self.calculate_metrics(y_true, pred_p, interval)
                    leaderboard.append({'Interval': interval, 'Model': 'Prophet', 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE (%)': mape})
                    df_preds['Prophet'] = pred_p
                    print(f"   ✅ Prophet: MAPE={mape:.2f}%")
                else:
                    print(f"   ⚠️ Thiếu Prophet: {model_path.name}")
            except Exception as e: print(f"   ❌ Prophet Error: {e}")

            # --- LOAD SCALER & AUTO FIX MISMATCH ---
            scaler_X_path = self.models_dir / f"scaler_X_{interval}.pkl"
            scaler_y_path = self.models_dir / f"scaler_y_{interval}.pkl"
            
            if not scaler_X_path.exists() or not scaler_y_path.exists():
                print(f"   ❌ Thiếu file Scaler. Bỏ qua XGB/LSTM.")
                continue

            try:
                scaler_X = joblib.load(scaler_X_path)
                scaler_y = joblib.load(scaler_y_path)
                
                # [FIX THÔNG MINH] Kiểm tra số lượng feature
                expected_features = scaler_X.n_features_in_
                current_features = len(feature_cols)
                
                if current_features != expected_features:
                    print(f"   ⚠️ Cảnh báo: Scaler cần {expected_features} cột, nhưng tìm thấy {current_features} cột.")
                    # Nếu thiếu 12 vs 20 -> Có khả năng 12 cột đầu là 12 cột cũ
                    # Ta sẽ thử cắt lấy đúng số lượng cột đầu tiên
                    print(f"   🔧 Đang tự động cắt {expected_features} cột đầu tiên để khớp...")
                    X_vals = df_test[feature_cols].values[:, :expected_features]
                else:
                    X_vals = df_test[feature_cols].values

                X_scaled = scaler_X.transform(X_vals)
                
            except Exception as e:
                print(f"   ❌ Lỗi Scaler không thể cứu chữa: {e}")
                print("   👉 Hãy chạy lại 'python -m scripts.train' để đồng bộ model.")
                continue

            # --- MODEL 2: XGBOOST ---
            try:
                model_path = self.models_dir / f"xgboost_{interval}.pkl"
                if model_path.exists():
                    model_xgb = joblib.load(model_path)
                    pred_sc = model_xgb.predict(X_scaled)
                    pred_xgb = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
                    
                    rmse, mse, mae, mape = self.calculate_metrics(y_true, pred_xgb, interval)
                    leaderboard.append({'Interval': interval, 'Model': 'XGBoost', 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE (%)': mape})
                    df_preds['XGBoost'] = pred_xgb
                    print(f"   ✅ XGBoost: MAPE={mape:.2f}%")
            except Exception as e: print(f"   ❌ XGBoost Error: {e}")

            # --- MODEL 3: LSTM ---
            try:
                model_path = self.models_dir / f"lstm_{interval}.pth"
                if model_path.exists():
                    input_dim = X_scaled.shape[1]
                    forecaster = LSTMForecaster(self.config, input_dim)
                    forecaster.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    forecaster.model.eval()
                    
                    n_lags = self.config['models']['lstm'].get('n_lags', 30)
                    if len(X_scaled) > n_lags:
                        X_seq = np.array([X_scaled[i:i+n_lags] for i in range(len(X_scaled)-n_lags)])
                        inp = torch.from_numpy(X_seq).float().to(DEVICE)
                        with torch.no_grad():
                            p = forecaster.model(inp).cpu().numpy().flatten()
                        
                        pred_lstm = scaler_y.inverse_transform(p.reshape(-1, 1)).flatten()
                        
                        y_trim = y_true[n_lags:]
                        rmse, mse, mae, mape = self.calculate_metrics(y_trim, pred_lstm, interval)
                        
                        leaderboard.append({'Interval': interval, 'Model': 'LSTM', 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE (%)': mape})
                        df_preds['LSTM'] = np.concatenate([[np.nan]*n_lags, pred_lstm])
                        print(f"   ✅ LSTM: MAPE={mape:.2f}%")
            except Exception as e: print(f"   ❌ LSTM Error: {e}")

            # Lưu kết quả
            out_csv = self.results_dir / f"predictions_{interval}.csv"
            df_preds.to_csv(out_csv, index=False)
            print(f"   💾 Saved CSV: {out_csv.name}")

        # --- XUẤT BẢNG KẾT QUẢ ---
        if leaderboard:
            df = pd.DataFrame(leaderboard).sort_values(by=['Interval', 'RMSE'])
            df = df[['Interval', 'Model', 'RMSE', 'MSE', 'MAE', 'MAPE (%)']]
            
            print(f"\n{'='*60}")
            print(f"🏆 BẢNG XẾP HẠNG KẾT QUẢ (LEADERBOARD)")
            print(f"{'='*60}")
            print(df.to_string(index=False))
            
            df.to_csv(self.results_dir / "final_leaderboard.csv", index=False)
        else:
            print("\n⚠️ Không có kết quả nào. Hãy kiểm tra lại thư mục saved_models!")

if __name__ == "__main__":
    Evaluator().run()