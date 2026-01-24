"""
MODULE: DATA PREPROCESSING PIPELINE """
import re
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 1. CONFIGURATION 
# ============================================================
# Thiết lập Logging để theo dõi tiến độ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Các hằng số quan trọng
# Khoảng thời gian bão (Downtime) đã được xác định từ EDA
DOWNTIME_START = pd.Timestamp("1995-08-01 14:52:01")
DOWNTIME_END = pd.Timestamp("1995-08-03 04:36:13")


# ============================================================
# 2. CLASS: PARSER 
# ============================================================
class LogParser:
    """
    Class chịu trách nhiệm đọc file thô và chuyển thành DataFrame.
    Sử dụng kỹ thuật đọc từng dòng để tránh tràn RAM.
    """
    
    # Regex Pattern: "Compiled" sẵn để chạy nhanh hơn
    # Bóc tách: Host, Timestamp, Request, Status, Bytes
    LOG_PATTERN = re.compile(
        r'(?P<host>\S+) \S+ \S+ '
        r'\[(?P<timestamp>[^\]]+)\] '
        r'"(?P<request>[^"]*)" '
        r'(?P<status>\d{3}) '
        r'(?P<bytes>\S+)'
    )

    def parse_line(self, line):
        """Xử lý từng dòng log một."""
        match = self.LOG_PATTERN.search(line)
        if not match:
            return None
        
        data = match.groupdict()
        
        # Xử lý Bytes: Ký tự '-' nghĩa là 0 bytes
        data['bytes'] = 0 if data['bytes'] == '-' else int(data['bytes'])
        
        return data

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Đọc file log và trả về DataFrame thô.
        Args:
            file_path (Path): Đường dẫn đến file data.
        """
        records = []
        logger.info(f"Đang đọc dữ liệu từ: {file_path}")
        
        try:
            # Dùng 'utf-8' và 'errors=ignore' để xử lý các ký tự lạ trong log cũ
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    parsed = self.parse_line(line)
                    if parsed:
                        records.append(parsed)
                    
                    # Log tiến độ mỗi 200,000 dòng để biết code không bị treo
                    if i > 0 and i % 200000 == 0:
                        logger.info(f"Đã xử lý {i} dòng...")
                        
            logger.info(f"Hoàn tất đọc file. Tổng số dòng hợp lệ: {len(records)}")
            return pd.DataFrame(records)
            
        except FileNotFoundError:
            logger.error(f"LỖI: Không tìm thấy file tại {file_path}")
            raise

# ============================================================
# 3. CLASS: PROCESSOR 
# ============================================================
class DataProcessor:
    """
    Class chịu trách nhiệm làm sạch, chuẩn hóa và gom nhóm (Aggregation).
    """
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa kiểu dữ liệu."""
        logger.info("Đang chuẩn hóa dữ liệu và xử lý múi giờ...")
        
        # 1. Chuyển đổi Timestamp có múi giờ
        df['timestamp'] = pd.to_datetime(
            df['timestamp'], 
            format='%d/%b/%Y:%H:%M:%S %z', 
            errors='coerce'
        )
        
        # Bỏ thông tin múi giờ để so sánh dễ dàng
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
       
        # 2. Ép kiểu số
        df['status'] = df['status'].astype(int)
        df['bytes'] = df['bytes'].astype(float) 
        
        # 3. Sắp xếp theo thời gian 
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df

    def aggregate_data(self, df: pd.DataFrame, window='5min') -> pd.DataFrame:
        """
        Gom nhóm dữ liệu theo khung thời gian (Resampling).
        Đây là bước chuẩn bị data cho Model Prophet/XGBoost.
        """
        logger.info(f"Đang gom nhóm dữ liệu theo khung {window}...")
        
        # Set timestamp làm index để resample
        df_indexed = df.set_index('timestamp')
        
        # Logic Aggregation:
        # - hits: Đếm tổng số yêu cầu
        # - bytes: Tính tổng và trung bình băng thông
        # - error_4xx: Phát hiện truy cập rác hoặc link hỏng 
        # - error_5xx: Theo dõi tình trạng quá tải của máy chủ 
        agg_df = df_indexed.resample(window).agg({
            'request': 'count',                 # Tổng lượt truy cập (Hits)
            'bytes': ['sum', 'mean'],           # Tổng và trung bình băng thông
            'status': [
                ('error_4xx', lambda x: ((x >= 400) & (x < 500)).sum()), # Lỗi do khách hàng/Bot
                ('error_5xx', lambda x: (x >= 500).sum())                # Lỗi do hệ thống quá tải
            ]
        })
        
        # Làm phẳng MultiIndex thành cột đơn
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df.rename(columns={
            'request_count': 'y',               # Biến mục tiêu cho dự báo lưu lượng
            'status_error_4xx': 'error_4xx',
            'status_error_5xx': 'error_5xx',
            'bytes_sum': 'total_bytes',
            'bytes_mean': 'avg_bytes'
        }, inplace=True)
        
        # Reset chỉ mục và chuẩn hóa tên cột thời gian sang 'ds'
        agg_df = agg_df.reset_index().rename(columns={'timestamp': 'ds'})
        
        # Xử lý Downtime 
        # Gắn cờ 1 nếu nằm trong vùng bão, 0 nếu bình thường
        agg_df['is_downtime'] = ((agg_df['ds'] >= DOWNTIME_START) & (agg_df['ds'] <= DOWNTIME_END)).astype(int)
        
        # Fill NA bằng 0 (cho những khoảng thời gian không có request nào)
        agg_df = agg_df.fillna(0)
        
        return agg_df

# ============================================================
# 4. MAIN EXECUTION
# ============================================================
def run_full_pipeline(file_type='train'):
    """
    Hàm Wrapper chạy một lần, xuất ra cả 3 khung thời gian: 1m, 5m, 15m.
    """
    # 1. Thiết lập đường dẫn tương đối 
    BASE_DIR = Path(__file__).resolve().parent.parent 
    DATA_DIR = BASE_DIR / "data"
    filename = "train.txt" if file_type == 'train' else "test.txt"
    file_path = DATA_DIR / filename
    
    # 2. Khởi tạo
    parser = LogParser()
    processor = DataProcessor()
    
    # 3. Chạy Pipeline (Chỉ load và clean 1 lần duy nhất để tiết kiệm RAM)
    raw_df = parser.load_data(file_path)
    clean_df = processor.clean_dataframe(raw_df)
    
    # 4. Aggregate cho cả 3 khung thời gian
    intervals = ['1min', '5min', '15min']
    processed_package = {}
    
    for interval in intervals:
        logger.info(f">>> Processing window: {interval}")
        agg_df = processor.aggregate_data(clean_df, window=interval)
        processed_package[interval] = agg_df
        
        # Tự động lưu ra file CSV
        output_name = f"processed_{file_type}_{interval}.csv"
        agg_df.to_csv(DATA_DIR / output_name, index=False)
        logger.info(f"Đã lưu: {output_name}")
        
    return processed_package

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" HỆ THỐNG XỬ LÝ DỮ LIỆU AUTOSCALING ")
    print("="*50)
    
    # Danh sách các tập dữ liệu cần xử lý
    data_types = ['train', 'test']
    
    try:
        for dtype in data_types:
            print(f"\n>>> ĐANG BẮT ĐẦU XỬ LÝ TẬP: {dtype.upper()}")
            run_full_pipeline(file_type=dtype)
        
        print("\n" + "*"*50)
        print(" CHÚC MỪNG: TOÀN BỘ PIPELINE ĐÃ HOÀN THÀNH! ")
        print(" - Dữ liệu TRAIN: Sẵn sàng để huấn luyện model. ")
        print(" - Dữ liệu TEST : Sẵn sàng để kiểm thử và EDA. ")
        print(" Tất cả file CSV đã nằm gọn trong thư mục 'data/'. ")
        print("*"*50)
        
    except Exception as e:
        # Nếu lỗi ở tập nào, hệ thống sẽ báo chính xác lỗi đó
        logger.error(f"LỖI HỆ THỐNG TRONG QUÁ TRÌNH XỬ LÝ: {e}")
