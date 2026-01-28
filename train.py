"""
Kịch bản train cho Dataflow 2026 - Phân tích Tự động mở rộng
train Prophet, XGBoost, LSTM cho cường độ
Trên 3 khung thời gian: 1 phút, 5 phút, 15 phút
"""
import os
import sys
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import ProphetForecaster, XGBoostForecaster, LSTMForecaster, create_forecaster, load_config

# ================================================================
# 1. CẤU HÌNH
# ================================================================
# Thiết lập Logging để theo dõi tiến độ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================================================================
# 2. CÁC LỚP CHÍNH
# ================================================================

class DataflowTrainer:
    """
    Đường dẫn train hoàn chỉnh cho bài toán Dataflow 2026
    """
    
    def __init__(self, data_dir='data', models_dir='models', results_dir='results', config_path='config/config.yaml'):
        # Lấy thư mục nơi tập lệnh này được đặt
        script_dir = Path(__file__).parent.absolute()
        
        # Chuyển đổi thành đường dẫn tuyệt đối nếu cung cấp đường dẫn tương đối
        self.data_dir = Path(data_dir) if Path(data_dir).is_absolute() else script_dir / data_dir
        self.models_dir = Path(models_dir) if Path(models_dir).is_absolute() else script_dir / models_dir
        self.results_dir = Path(results_dir) if Path(results_dir).is_absolute() else script_dir / results_dir
        self.config_path = Path(config_path) if Path(config_path).is_absolute() else script_dir / config_path
        
        # Tải cấu hình
        self.config = load_config(self.config_path)
        
        # Tạo thư mục
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Cấu hình
        self.time_windows = self.config.get('processing', {}).get('intervals', ['1min', '5min', '15min'])
        self.target_columns = ['intensity']
        self.model_names = ['prophet', 'xgboost', 'lstm']
        
        # Lưu trữ kết quả
        self.all_results = {}
    
    def load_data(self, time_window, split='train'):
       
        if split == 'train':
            filename = f"prepared_{split}_{time_window}.csv"
        else:
            filename = f"prepared_{split}_{time_window}.csv"
        
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.error(f"File không tồn tại: {filepath} -- Bỏ qua khung thời gian {time_window}.")
            return None
        
        logger.info(f"Load {filepath}")
        df = pd.read_csv(filepath)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.set_index('ds')

        df = df.sort_index()
        logger.info(f"{df.shape[0]} obs, {df.shape[1]} cols")
        
        return df
    
    def train_single_model(self, model_name, train_df, train_series, target_col, time_window):
        
        logger.info(f"  Đang train {model_name.upper()}...")
        
        try:
            # Tải tham số từ config
            model_cfg = self.config.get('models', {}).get(model_name, {})
            
            if not model_cfg.get('enabled', False):
                logger.warning(f"    {model_name.upper()} bị vô hiệu hóa trong config")
                return None
            
            # Tạo mô hình với các tham số config
            model = create_forecaster(model_name, config_path=self.config_path)
            
            # Train - Prophet dùng series, XGBoost/LSTM dùng DataFrame
            if model_name == 'prophet':
                success = model.fit(train_series)
            else:
                success = model.fit(train_df, target_col=target_col)
            
            if success:
                logger.info(f"    ✓ {model_name.upper()} được train thành công")
                return model
            else:
                logger.warning(f"    ✗ Không thể train {model_name.upper()}")
                return None
        
        except Exception as e:
            logger.error(f"    ✗ Lỗi khi train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_model(self, model, test_df, test_series, model_name):
        
        try:
            if model_name == 'xgboost':
                # XGBoost supports test_df for vectorized prediction
                predictions = model.predict(test_df=test_df, steps=len(test_df))
            elif model_name == 'lstm':
                # LSTM also supports test_df for faster vectorized prediction
                predictions = model.predict(test_df=test_df, steps=len(test_df))
            else:
                predictions = model.predict(steps=len(test_series))
            
            if predictions is None:
                logger.warning(f"  Không thể tạo dự báo")
                return None
            
            pred_len = int(len(predictions))
            logger.info(f"  Predictions: {pred_len}")
            
            if hasattr(test_series, 'values'):
                y_true = test_series.values[:pred_len]
            else:
                y_true = test_series[:pred_len]
            
            metrics = model.evaluate(y_true, predictions)
            
            return {
                'metrics': metrics,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                'actual': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
            }
        
        except Exception as e:
            logger.error(f"  Lỗi khi đánh giá mô hình: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def train_for_window_and_target(self, time_window, target_col):
        """
        train tất cả các mô hình cho một khung thời gian cụ thể và mục tiêu
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"KHUNG THỜI GIAN: {time_window} | MỤC TIÊU: {target_col}")
        logger.info(f"{'='*70}")
        
        # Tải dữ liệu
        train_df = self.load_data(time_window, split='train')
        test_df = self.load_data(time_window, split='test')
        
        if train_df is None or test_df is None:
            logger.error(f"Không thể tải dữ liệu cho {time_window}")
            return
        
        # Kiểm tra xem cột mục tiêu có tồn tại không
        if target_col not in train_df.columns:
            logger.error(f"Cột mục tiêu '{target_col}' không tìm thấy")
            return
        
        # Trích xuất chuỗi mục tiêu
        train_series = train_df[target_col]
        test_series = test_df[target_col]
        
        logger.info(f"\nTrain stats - Mean: {train_series.mean():.2f}, Std: {train_series.std():.2f}, Min: {train_series.min():.2f}, Max: {train_series.max():.2f}")
        
        # train tất cả các mô hình
        results = {}
        trained_models = {}
        
        for model_name in self.model_names:
            model = self.train_single_model(
                model_name, 
                train_df,
                train_series, 
                target_col, 
                time_window
            )
            
            if model is not None:
                trained_models[model_name] = model
                
                # Đánh giá trên tập kiểm tra
                logger.info(f"  Đánh giá {model_name.upper()}...")
                eval_results = self.evaluate_model(model, test_df, test_series, model_name)
                
                if eval_results is not None:
                    results[model_name] = eval_results
                    
                    # Ghi nhật ký các chỉ số
                    metrics = eval_results['metrics']
                    logger.info(f"    RMSE: {metrics['rmse']:.4f}")
                    logger.info(f"    MAE:  {metrics['mae']:.4f}")
                    logger.info(f"    MAPE: {metrics['mape']:.2f}%")
                    logger.info(f"    MSE:  {metrics['mse']:.4f}")
                
                # Lưu mô hình
                model_filename = f"{target_col}_{time_window}_{model_name}.pkl"
                model_path = self.models_dir / model_filename
                try:
                    model.save(str(model_path))
                    logger.info(f"    Mô hình được lưu vào {model_path}")
                except Exception as e:
                    logger.warning(f"    Không thể lưu mô hình: {e}")
        
        # Lưu kết quả
        key = f"{target_col}_{time_window}"
        self.all_results[key] = results
        
        # Tạo báo cáo so sánh
        self.generate_comparison_report(results, time_window, target_col)
        
        return results
    
    def generate_comparison_report(self, results, time_window, target_col):
        """Tạo bảng so sánh cho các mô hình"""
        if not results:
            return
        
        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL COMPARISON - {time_window} - {target_col}")
        logger.info(f"{'='*70}")
        
        # Tạo bảng so sánh
        header = f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'MAPE (%)':<12} {'MSE':<12}"
        logger.info(header)
        logger.info("-" * 70)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            row = (f"{model_name.upper():<15} "
                   f"{metrics['rmse']:<12.4f} "
                   f"{metrics['mae']:<12.4f} "
                   f"{metrics['mape']:<12.2f} "
                   f"{metrics['mse']:<12.4f}")
            logger.info(row)
        
        # Tìm mô hình tốt nhất cho từng chỉ số
        logger.info("\nCác mô hình tốt nhất:")
        best_rmse = min(results.items(), key=lambda x: x[1]['metrics']['rmse'])
        best_mae = min(results.items(), key=lambda x: x[1]['metrics']['mae'])
        best_mape = min(results.items(), key=lambda x: x[1]['metrics']['mape'])
        
        logger.info(f"  RMSE tốt nhất: {best_rmse[0].upper()} ({best_rmse[1]['metrics']['rmse']:.4f})")
        logger.info(f"  Best MAE: {best_mae[0].upper()} ({best_mae[1]['metrics']['mae']:.4f})")
        logger.info(f"  MAPE tốt nhất: {best_mape[0].upper()} ({best_mape[1]['metrics']['mape']:.2f}%)")
    
    def run_complete_training(self):
        """
        Chạy đường dẫn train hoàn chỉnh cho tất cả các kết hợp
        """
        logger.info("\n" + "="*70)
        logger.info("DATAFLOW 2026 - ĐƯỜNG DẪN train HOÀN CHỈNH")
        logger.info("="*70)
        logger.info(f"Khung thời gian: {self.time_windows}")
        logger.info(f"Cột mục tiêu: {self.target_columns}")
        logger.info(f"Mô hình: {self.model_names}")
        logger.info(f"Cấu hình: {self.config_path}")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # train cho tất cả các kết hợp
        for time_window in self.time_windows:
            for target_col in self.target_columns:
                try:
                    self.train_for_window_and_target(time_window, target_col)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý {time_window} - {target_col}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Lưu tất cả các kết quả
        self.save_all_results()
        
        # Tạo tóm tắt
        self.generate_summary_report()
        
        # In tóm tắt cuối cùng
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*70)
        logger.info(" HOÀN THÀNH")
        logger.info("="*70)
        logger.info(f"Thời gian: {duration}")
        logger.info(f"Kết quả được lưu tại: {self.results_dir}")
        logger.info(f"Mô hình được lưu tại: {self.models_dir}")
        logger.info("="*70)
    
    def save_all_results(self):
        """Lưu tất cả các kết quả vào tệp JSON"""
        results_file = self.results_dir / 'all_metrics.json'
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.all_results, f, indent=2)
            logger.info(f"\nTất cả các kết quả được lưu vào {results_file}")
        except Exception as e:
            logger.error(f"Không thể lưu kết quả: {e}")
    
    def generate_summary_report(self):
        """Tạo báo cáo tóm tắt của tất cả các thử nghiệm"""
        report_file = self.results_dir / 'summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DATAFLOW 2026 - TRAINING SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            for key, results in self.all_results.items():
                if '_' in key:
                    parts = key.rsplit('_', 1)
                    target = parts[0] if len(parts) > 1 else key
                    window = parts[1] if len(parts) > 1 else ''
                else:
                    target = key
                    window = ''
                
                f.write(f"\n{target.upper()} - {window}\n")
                f.write("-"*70 + "\n")
                f.write(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'MAPE (%)':<12}\n")
                f.write("-"*70 + "\n")
                
                for model_name, result in results.items():
                    metrics = result['metrics']
                    f.write(f"{model_name.upper():<15} "
                           f"{metrics['rmse']:<12.4f} "
                           f"{metrics['mae']:<12.4f} "
                           f"{metrics['mape']:<12.2f}\n")
                f.write("\n")
        
        logger.info(f"Summary report saved to {report_file}")


# ================================================================
# 3. HÀM MAIN
# ================================================================

def main():
    """Điểm nhập liệu chính"""
    import argparse
    
    parser = argparse.ArgumentParser(description=' các mô hình dự báo cho Dataflow 2026')
    parser.add_argument('--data-dir', type=str, default='data', help='Thư mục dữ liệu')
    parser.add_argument('--models-dir', type=str, default='models', help='Thư mục đầu ra của mô hình')
    parser.add_argument('--results-dir', type=str, default='results', help='Thư mục đầu ra của kết quả')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Đường dẫn tệp cấu hình')
    
    args = parser.parse_args()
    
    # Tạo trainer
    trainer = DataflowTrainer(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        config_path=args.config
    )
    
    # Chạy 
    trainer.run_complete_training()


# ================================================================
# 4. ĐIỂM VÀO
# ================================================================

if __name__ == "__main__":
    main()
