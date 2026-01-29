"""
MODULE: FEATURE ENGINEERING & FILTERING PIPELINE
Xử lý sau khi EDA hoàn thành:
1. Lọc downtime (dựa trên EDA insights)
2. Lọc bot/DDoS (dựa trên error_rate threshold)
3. Tạo Lag features (24h và 7d cycles)
4. Tạo Time features (hour, day_of_week, is_weekend)
"""
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Tải cấu hình từ file YAML."""
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# ============================================================
# CLASS: FEATURE ENGINEER
# ============================================================
class FeatureEngineer:
    """
    Xử lý lọc dữ liệu và tạo features sau khi EDA phát hiện insights.
    """
    
    def __init__(self, config):
        self.config = config
       
        self.downtime_start = pd.Timestamp("1995-07-28 13:35:00")
        self.downtime_end = pd.Timestamp("1995-08-03 04:37:00")
        self.bot_error_threshold = config.get('analysis', {}).get('bot_error_threshold', 0.8)
    
    def filter_downtime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lọc bỏ giai đoạn downtime (từ EDA insights).
        
        Downtime: Từ ngày 28/07 13:35:00 -> 03/08 04:36:13
        """
        logger.info(f"🔍 Đang lọc downtime: {self.downtime_start} → {self.downtime_end}")
        
        before_count = len(df)
        
        # Lọc bỏ khoảng downtime
        df_filtered = df[~((df['ds'] >= self.downtime_start) & (df['ds'] <= self.downtime_end))].copy()
        
        removed_count = before_count - len(df_filtered)
        logger.info(f"  ✓ Đã loại bỏ {removed_count:,} observation trong khoảng downtime")
        
        return df_filtered
    
    def filter_bot_ddos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lọc bỏ các observation có tỉ lệ lỗi bất thường (phát hiện Bot/DDoS).
        """
        logger.info(f"🔍 Đang lọc bot/DDoS (error_rate >= {self.bot_error_threshold})")
        
        before_count = len(df)
        df_filtered = df[df['error_rate'] < self.bot_error_threshold].copy()
        removed_count = before_count - len(df_filtered)
        
        logger.info(f"  ✓ Đã loại bỏ {removed_count:,} observation (noise)")
        
        return df_filtered

    def filter_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lọc các giá trị intensity không hợp lệ hoặc ngoại lai.

        - Loại NaN trong cột `intensity`
        - Loại giá trị âm
        - Loại các ngoại lai trên percentile cao (mặc định 99.5%)
        """
        logger.info("🔍 Đang lọc intensity (NaN, âm, ngoại lai)")

        if 'intensity' not in df.columns:
            logger.warning("  ⚠️ Không tìm thấy cột 'intensity' — bỏ qua bước lọc intensity")
            return df

        before = len(df)

        # loại NaN và giá trị âm
        df_filtered = df.dropna(subset=['intensity']).copy()
        df_filtered = df_filtered[df_filtered['intensity'] >= 0]

        # loại ngoại lai phía trên (upper percentile)
        try:
            upper_pct = 0.995
            upper_val = df_filtered['intensity'].quantile(upper_pct)
            outliers = df_filtered[df_filtered['intensity'] > upper_val]
            if len(outliers) > 0:
                logger.info(f"  ✓ Loại {len(outliers):,} observation > {int(upper_pct*100)}th percentile (>{upper_val:.2f})")
                df_filtered = df_filtered[df_filtered['intensity'] <= upper_val]
        except Exception:
            logger.exception("  ⚠️ Không thể tính percentile intensity — bỏ qua lọc ngoại lai")

        removed = before - len(df_filtered)
        logger.info(f"  ✓ Đã loại bỏ {removed:,} observation (intensity)")

        return df_filtered

    
    
    def create_lag_features(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Tạo Lag features cho 24h và 7d cycles.
        
        Args:
            df: DataFrame sau lọc
            interval: '1min', '5min', hoặc '15min'
        """
        logger.info(f"🔧 Tạo Lag features cho khung {interval}...")
        
        if interval == '1min':
            df['lag_1440'] = df['intensity'].shift(1440)      # 24h (1440 × 1min)
            df['lag_10080'] = df['intensity'].shift(10080)     # 7d (10080 × 1min)
            logger.info(f"  ✓ lag_1440 (24h), lag_10080 (7d)")
        
        elif interval == '5min':
            df['lag_288'] = df['intensity'].shift(288)         # 24h (288 × 5min)
            df['lag_2016'] = df['intensity'].shift(2016)       # 7d (2016 × 5min)
            logger.info(f"  ✓ lag_288 (24h), lag_2016 (7d)")
        
        elif interval == '15min':
            df['lag_96'] = df['intensity'].shift(96)           # 24h (96 × 15min)
            df['lag_672'] = df['intensity'].shift(672)         # 7d (672 × 15min)
            logger.info(f"  ✓ lag_96 (24h), lag_672 (7d)")
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo Time features từ timestamp.
        Hỗ trợ cả cột 'ds' và 'timestamp'.
        """
        logger.info("🔧 Tạo Time features (hour, day_of_week, is_weekend)...")
        
        # Xác định cột thời gian
        time_col = None
        if 'ds' in df.columns:
            time_col = 'ds'
        elif 'timestamp' in df.columns:
            time_col = 'timestamp'
        else:
            logger.warning("  ⚠️ Không tìm thấy cột timestamp/ds")
            return df
        
        # Đảm bảo là datetime
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Hour: 0-23
        df['hour'] = df[time_col].dt.hour
        
        # Day of week: 0-6 (Monday=0, Sunday=6)
        df['day_of_week'] = df[time_col].dt.dayofweek
        
        # Is weekend: 1 nếu Saturday-Sunday, 0 nếu weekday
        df['is_weekend'] = (df[time_col].dt.dayofweek >= 5).astype(int)
        
        logger.info(f"  ✓ hour, day_of_week, is_weekend")
        
        return df
    
    def process_single_interval(self, input_path: Path, output_path: Path, interval: str, file_type: str):
        """
        Xử lý một khung thời gian duy nhất.
        
        Args:
            input_path: Đường dẫn file RAW (từ preprocessing)
            output_path: Đường dẫn file output (sau feature engineering)
            interval: '1min', '5min', '15min'
            file_type: 'train' hoặc 'test'
        """
        logger.info(f"▶ Đang xử lý {file_type.upper()} - {interval}...")

        if not input_path.exists():
            logger.error(f"  ❌ File không tồn tại: {input_path}")
            return False

        # Load main input (raw, chưa lọc) and mark source
        df_main = pd.read_csv(input_path)
        df_main['ds'] = pd.to_datetime(df_main['ds'])
        df_main = df_main.copy()
        df_main['__source__'] = 'train' if file_type == 'train' else 'test'
        logger.info(f"  Loaded {len(df_main):,} observations from input")

        combined = df_main

        # nếu là test, load thêm train để tạo lag features chính xác
        train_input = input_path.parent / f"processed_train_{interval}.csv"
        if file_type == 'test' and train_input.exists():
            train_df_raw = pd.read_csv(train_input)
            train_df_raw['ds'] = pd.to_datetime(train_df_raw['ds'])
            train_df_raw = train_df_raw.copy()
            train_df_raw['__source__'] = 'train'
            combined = pd.concat([train_df_raw, df_main], ignore_index=True)
            logger.info(f"  Loaded train ({len(train_df_raw):,}) and concatenated for lag calculation")

        # Sắp xếp theo thời gian trước khi tạo lag features
        combined = combined.sort_values('ds').reset_index(drop=True)
        
        # NOTE: Lag features được tính từ test data thật (khi file_type='test')
        # Điều này có vẻ như data leakage, NHƯNG:
        # - Đây là ở feature engineering time (offline), không phải prediction time
        # - Khi predict (online), XGBoostForecaster/LSTMForecaster.predict() sẽ cập nhật
        #   lag features từ PREDICTIONS (không phải test data thật) via roll-forward mechanism
        # - Xem models.py predict() method để hiểu chi tiết
        combined = self.create_lag_features(combined, interval)
        combined = self.create_time_features(combined)

        # Sau khi tạo lag features, lọc dữ liệu
        combined = self.filter_downtime(combined)
        combined = self.filter_bot_ddos(combined)
        combined = self.filter_intensity(combined)

        # Tách lại train/test và xuất
        if file_type == 'test':
            out_df = combined[combined['__source__'] == 'test'].copy().reset_index(drop=True)
            logger.info(f"  Extracted test data: {len(out_df):,} observations")
        else:
            out_df = combined[combined['__source__'] == 'train'].copy().reset_index(drop=True)
            out_df = out_df.dropna().reset_index(drop=True)
            logger.info(f"  ✓ Sau xử lý: {len(out_df):,} observations")

        if '__source__' in out_df.columns:
            out_df = out_df.drop(columns=['__source__'])

        out_df.to_csv(output_path, index=False)
        logger.info(f"  ✅ Xuất: {output_path}")

        return True

# ============================================================
# MAIN EXECUTION
# ============================================================
def run_feature_engineering(file_type='train'):
    """
    Chạy feature engineering cho train hoặc test.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FEATURE ENGINEERING: {file_type.upper()}")
    logger.info(f"{'='*60}")

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    engineer = FeatureEngineer(CONFIG)
    intervals = CONFIG['processing']['intervals']

    for interval in intervals:
        input_file = f"processed_{file_type}_{interval}.csv"
        output_file = f"prepared_{file_type}_{interval}.csv"

        input_path = DATA_DIR / input_file
        output_path = DATA_DIR / output_file

        try:
            success = engineer.process_single_interval(
                input_path,
                output_path,
                interval,
                file_type
            )
            if not success:
                logger.error(f"  Failed to process {interval}")
        except Exception as e:
            logger.error(f"  ❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" FEATURE ENGINEERING & FILTERING PIPELINE")
    print("="*70)

    try:
        run_feature_engineering(file_type='train')
        run_feature_engineering(file_type='test')

        print("\n" + "*"*70)
        print(" ✅ HOÀN THÀNH FEATURE ENGINEERING!")
        print(" 📁 Các file đã tạo:")
        print("    - prepared_train_1min.csv, prepared_train_5min.csv, prepared_train_15min.csv")
        print("    - prepared_test_1min.csv, prepared_test_5min.csv, prepared_test_15min.csv")
        print("*"*70 + "\n")

    except Exception as e:
        logger.error(f"❌ LỖI HỆ THỐNG: {e}")
        import traceback
        traceback.print_exc()
