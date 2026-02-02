"""
SCRIPT: FEATURE ENGINEERING RUNNER
----------------------------------
"""
import sys
import yaml
import logging
import pandas as pd
from pathlib import Path

# Setup Path để import được src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Class từ src 
from src.feature_engineering import FeatureEngineeringPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    p = PROJECT_ROOT / "config/config.yaml"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f)
    sys.exit("❌ Config not found")

def main():
    config = load_config()
    data_dir = PROJECT_ROOT / "data"
    pipeline = FeatureEngineeringPipeline(config)
    
    target_intervals = ['1min', '5min', '15min']
    train_context_cache = {} # Cache 8 ngày cuối của train để dùng cho test

    print(f"\n🚀 START FEATURE ENGINEERING \n")

    # 1. Xử lý TRAIN trước
    for interval in target_intervals:
        input_path = data_dir / f"processed_train_{interval}.csv"
        if not input_path.exists(): continue
        
        logger.info(f"▶ Processing TRAIN: {interval}")
        df = pd.read_csv(input_path)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Gọi hàm xử lý từ src
        df_prepared = pipeline.process(df, interval)
        
        # Lưu context cho bước Test
        cutoff = df_prepared['ds'].max() - pd.Timedelta(days=8)
        train_context_cache[interval] = df_prepared[df_prepared['ds'] > cutoff][['ds', 'intensity']].copy()
        
        # Lưu file
        out_path = data_dir / f"prepared_train_{interval}.csv"
        df_prepared.to_csv(out_path, index=False)
        logger.info(f"   💾 Saved: {out_path.name} (Cols: {len(df_prepared.columns)})")

    # 2. Xử lý TEST sau
    for interval in target_intervals:
        input_path = data_dir / f"processed_test_{interval}.csv"
        if not input_path.exists(): continue
        
        logger.info(f"▶ Processing TEST: {interval}")
        df = pd.read_csv(input_path)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Lấy context tương ứng
        context = train_context_cache.get(interval)
        
        # Gọi hàm xử lý
        df_prepared = pipeline.process(df, interval, context_df=context)
        
        # Lưu file
        out_path = data_dir / f"prepared_test_{interval}.csv"
        df_prepared.to_csv(out_path, index=False)
        logger.info(f"   💾 Saved: {out_path.name} (Cols: {len(df_prepared.columns)})")

if __name__ == "__main__":
    main()