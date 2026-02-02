"""
MODULE: Feature Engineering Pipeline
-------------------------------------------------------------------
Mô tả:
    Chuyển đổi dữ liệu chuỗi thời gian thô thành 
    Feature Matrix để phục vụ huấn luyện mô hình học máy.

Các kỹ thuật áp dụng:
    1. Time Continuity Restoration: Tái tạo trục thời gian liên tục để xử lý các khoảng trống 
       do quá trình lọc nhiễu gây ra.
    2. Cyclical Encoding: Mã hóa lượng giác (Sin/Cos) cho đặc trưng thời gian (Giờ, Thứ).
    3. Rolling Statistics: Tính toán xu hướng trượt để bắt tín hiệu Trend/Volatility.
    4. Dynamic Lagging: Tự động tính toán bước trễ dựa trên tần suất dữ liệu .

"""

import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
import sys

# Tắt cảnh báo FutureWarnings
warnings.filterwarnings('ignore')

# CẤU HÌNH LOGGING & CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config():
    """
    Tải cấu hình hệ thống từ file YAML với cơ chế tự động dò tìm đường dẫn.
    Đảm bảo tính linh hoạt khi chạy script từ các thư mục khác nhau.
    """
    search_paths = [
        Path("config/config.yaml"), 
        Path("../config/config.yaml"), 
        Path("../../config/config.yaml")
    ]
    for p in search_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    
    logger.error("❌ CRITICAL: Không tìm thấy tệp cấu hình 'config.yaml'.")
    sys.exit(1)

CONFIG = load_config()

# CLASS XỬ LÝ TRUNG TÂM
class FeatureEngineeringPipeline:
    def __init__(self, config):
        self.config = config
        
    def _parse_interval_minutes(self, interval_str):
        """
        Phân tích chuỗi định dạng '5min', '15min' sang số nguyên phút.
        Hỗ trợ tính toán số bước nhảy cho Lag Features.
        """
        match = re.match(r"(\d+)", interval_str)
        return int(match.group(1)) if match else 5

    def _restore_time_continuity(self, df, interval_str):
        """
        Khôi phục tính liên tục của thời gian (Time Index Reconstruction).
        
        Vấn đề: Dữ liệu sau EDA bị cắt bỏ một khoảng bão/lỗi. Nếu dùng shift() trực tiếp,
        Model sẽ học sai quy luật (nhìn nhầm dữ liệu của 5 ngày trước thành dữ liệu vừa xảy ra).
        
        Giải pháp: 
        1. Tạo một trục thời gian chuẩn đầy đủ .
        2. Reindex DataFrame vào trục này. Các khoảng trống sẽ được lấp đầy bằng NaN.
        3. Khi tính toán Lag, shift() sẽ gặp NaN -> Lag chính xác về mặt vật lý.
        """
        if 'ds' not in df.columns:
            raise ValueError("DataFrame thiếu cột 'ds' (Timestamp).")

        df = df.set_index('ds').sort_index()
        
        # Xác định tần suất chuẩn
        minutes = self._parse_interval_minutes(interval_str)
        freq = f"{minutes}T" # Ví dụ: '1T', '5T', '15T'
        
        # Tạo trục thời gian liên tục từ điểm đầu đến điểm cuối
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        
        # Reindex: Các điểm bị thiếu sẽ sinh ra dòng mới với giá trị NaN
        df_restored = df.reindex(full_idx)
        
        logger.debug(f"   🔧 Đã khôi phục trục thời gian: {len(df)} -> {len(df_restored)} dòng (Thêm {len(df_restored)-len(df)} khoảng trống).")
        
        # Reset index để trả lại cột 'ds'
        return df_restored.reset_index().rename(columns={'index': 'ds'})

    def generate_cyclical_features(self, df):
        """
        Mã hóa đặc trưng chu kỳ cho thời gian.
        
        Lý do: Máy học không hiểu tính tuần hoàn của giờ giấc (23h và 0h rất xa nhau về số học).
        Phép biến đổi Sin/Cos đưa chúng về gần nhau trên không gian vector.
        """
        df = df.copy()
        
        # Trích xuất thông tin thời gian
        df['hour'] = df['ds'].dt.hour
        df['day_of_week'] = df['ds'].dt.dayofweek
        
        # 1. Chu kỳ Ngày (24h)
        # Giúp model hiểu tải trọng đỉnh thường rơi vào trưa/chiều
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 2. Chu kỳ Tuần (7 ngày)
        # Giúp model phân biệt ngày thường và cuối tuần một cách mượt mà
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 3. Đặc trưng Cuối tuần (Boolean) 
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Loại bỏ các cột thô để giảm nhiễu 
        # df.drop(columns=['hour', 'day_of_week'], inplace=True)
        
        return df

    def generate_lag_rolling_features(self, df, interval_str):
        """
        Tạo đặc trưng chuỗi thời gian động theo khung thời gian.
        
        Tự động tính toán số bước dựa trên interval:
        - 1min: 1h = 60 steps
        - 5min: 1h = 12 steps
        - 15min: 1h = 4 steps
        """
        df = df.copy()
        target_col = 'intensity'
        
        # Tính toán số bước nhảy 
        minutes = self._parse_interval_minutes(interval_str)
        steps_per_hour = 60 // minutes
        steps_per_day = 24 * steps_per_hour
        steps_per_week = 7 * steps_per_day
        
        logger.info(f"    Cấu hình Feature cho {interval_str}: 1h={steps_per_hour} steps, 24h={steps_per_day} steps.")
        
        # A. LAG FEATURES 
        # -----------------------------------------------------------
        lags = {
            'lag_1step': 1,                 # Ngay trước đó
            'lag_1h': steps_per_hour,       # 1 giờ trước 
            'lag_24h': steps_per_day,       # Cùng giờ ngày hôm qua 
            'lag_7d': steps_per_week        # Cùng giờ tuần trước
        }
        
        for name, step in lags.items():
            df[name] = df[target_col].shift(step)
            
        # B. ROLLING FEATURES 
        # -----------------------------------------------------------
        # Cửa sổ quan sát: 4 giờ gần nhất
        window_size = steps_per_hour * 4 
        
        # Lưu ý QUAN TRỌNG: Phải Shift(1) trước khi Rolling để tránh Data Leakage 
        shifted = df[target_col].shift(1)
        
        df['roll_mean_4h'] = shifted.rolling(window=window_size).mean() # Xu hướng trung bình
        df['roll_std_4h'] = shifted.rolling(window=window_size).std()   # Độ biến động 
        df['roll_max_4h'] = shifted.rolling(window=window_size).max()   # Đỉnh tải cục bộ
        
        return df

    def cleanup_and_validate(self, df, target_col='intensity'):
        """
        Loại bỏ các dòng NaN sinh ra do Lag/Rolling hoặc do quá trình Reindex (Vùng Gap).
        """
        initial_len = len(df)
        
        # Chỉ giữ lại các dòng có Target hợp lệ 
        # (Tự động loại bỏ các dòng Gap NaN và các dòng đầu tiên chưa đủ dữ liệu để tính Lag)
        df_clean = df.dropna(subset=[target_col, 'lag_24h', 'roll_mean_4h']).copy()
        
        # Fill NaN còn sót lại (nếu có) bằng 0 để an toàn cho Model
        df_clean = df_clean.fillna(0)
        
        dropped = initial_len - len(df_clean)
        logger.info(f"   🧹 Đã dọn dẹp {dropped:,} dòng (NaN do Gap & Warm-up period).")
            
        return df_clean

    def execute(self):
        """
        Hàm điều phối chính.
        Thực hiện quy trình cho toàn bộ các khung thời gian được yêu cầu.
        """
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data"
        
        # Dictionary để lưu Context từ tập Train (dùng để nối vào đầu tập Test)
        train_context_cache = {}
        
        # Xử lý 3 khung thời gian
        target_intervals = ['1min', '5min', '15min']

        print("\n" + "="*70)
        print(" 🚀 FEATURE ENGINEERING ENGINE (PRO VERSION)")
        print(f"    Target Intervals: {target_intervals}")
        print("="*70)

        for dataset_type in ['train', 'test']:
            print(f"\n📂 Đang xử lý tập dữ liệu: {dataset_type.upper()}")
            
            for interval in target_intervals:
                # File đầu vào từ bước Preprocessing
                input_file = f"processed_{dataset_type}_{interval}.csv"
                # File đầu ra cho Training
                output_file = f"prepared_{dataset_type}_{interval}.csv"
                
                input_path = data_dir / input_file
                output_path = data_dir / output_file
                
                if not input_path.exists():
                    logger.warning(f"   ⚠️ Bỏ qua {interval}: Không tìm thấy file nguồn {input_file}")
                    # Nếu file 1min/15min chưa có, bỏ qua luôn
                    continue

                logger.info(f"▶ Bắt đầu xử lý: {interval}")

                # 1. Load Dữ liệu sạch
                df = pd.read_csv(input_path)
                df['ds'] = pd.to_datetime(df['ds'])
                
                # --- LOGIC: XỬ LÝ CONTEXT CHO TEST (TRÁNH MẤT DỮ LIỆU ĐẦU KỲ) ---
                original_test_start = None
                
                if dataset_type == 'test' and interval in train_context_cache:
                    original_test_start = df['ds'].min()
                    # Lấy context từ train nối vào trước test
                    context_df = train_context_cache[interval]
                    df = pd.concat([context_df, df], axis=0, ignore_index=True)
                    # Xóa trùng lặp nếu có 
                    df = df.drop_duplicates(subset=['ds']).sort_values('ds')

                # 2. Khôi phục tính liên tục (Chạy trên dữ liệu đã nối để lấp gap giữa train-test)
                df = self._restore_time_continuity(df, interval)
                
                # Nếu là Train, lưu lại 8 ngày cuối làm context cho Test (đủ cho lag_7d)
                if dataset_type == 'train':
                    cutoff_time = df['ds'].max() - pd.Timedelta(days=8)
                    train_context_cache[interval] = df[df['ds'] > cutoff_time].copy()
                # --------------------------------------------------------------------

                # 3. Tạo đặc trưng thời gian 
                df = self.generate_cyclical_features(df)
                
                # 4. Tạo đặc trưng chuỗi động 
                df = self.generate_lag_rolling_features(df, interval)
                
                # --- CẮT TRẢ VỀ ĐÚNG KÍCH THƯỚC TEST ---
                if dataset_type == 'test' and original_test_start is not None:
                    df = df[df['ds'] >= original_test_start].copy()
                
                # 5. Dọn dẹp & Kiểm tra
                df = self.cleanup_and_validate(df)
                
                # 6. Lưu kết quả
                df.to_csv(output_path, index=False)
                logger.info(f"   ✅ Hoàn tất: {output_file} | Shape: {df.shape}")

# ENTRY POINT
if __name__ == "__main__":
    try:
        pipeline = FeatureEngineeringPipeline(CONFIG)
        pipeline.execute()
        print("\n✅ [SUCCESS] QUY TRÌNH KỸ THUẬT ĐẶC TRƯNG HOÀN TẤT.")
    except Exception as e:
        logger.error(f"❌ [FAILURE] Hệ thống gặp lỗi: {e}")
        import traceback
        traceback.print_exc()