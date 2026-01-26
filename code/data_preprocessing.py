"""
MODULE: DATA PREPROCESSING PIPELINE """
import re
import yaml
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
def load_config():
    # Trỏ đến thư mục config và đọc file config.yaml
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
# Lấy thông tin downtime từ config
CONFIG = load_config()
DOWNTIME_START = pd.Timestamp(CONFIG['processing']['downtime']['start'])
DOWNTIME_END = pd.Timestamp(CONFIG['processing']['downtime']['end'])


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
        
       # --- TÍNH TOÁN CÁC BIẾN ĐẶC TRƯNG NÂNG CAO (FEATURE ENGINEERING) ---
        # 1. Tỉ lệ lỗi (Error Rate): Phản ánh độ ổn định và sức khỏe của hệ thống
        agg_df['error_rate'] = (agg_df['error_4xx'] + agg_df['error_5xx']) / (agg_df['y'] + 1e-8)
        
        # 2. Cường độ tải (Resource Intensity): Chỉ số tải thực tế dựa trên lưu lượng và băng thông
        # Sử dụng trọng số từ cấu hình để phản ánh áp lực lên tài nguyên phần cứng
        weight = CONFIG.get('analysis', {}).get('resource_weight', 1.0)
        agg_df['intensity'] = agg_df['y'] * agg_df['avg_bytes'] * weight

        # 3. Phân loại trạng thái hệ thống (Downtime Labeling)
        # Gắn nhãn các giai đoạn xảy ra sự cố dựa trên khung thời gian cấu hình
        agg_df['is_downtime'] = ((agg_df['ds'] >= DOWNTIME_START) & (agg_df['ds'] <= DOWNTIME_END)).astype(int)
        
        # Xử lý các giá trị thiếu bằng phương pháp điền số 0 để đảm bảo tính liên tục của dữ liệu
        return agg_df.fillna(0)

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
    filename = CONFIG['paths']['train_file'] if file_type == 'train' else CONFIG['paths']['test_file']
    file_path = (BASE_DIR / CONFIG['paths']['input_dir']) / filename
    
    # 2. Khởi tạo
    parser = LogParser()
    processor = DataProcessor()
    
    # 3. Chạy Pipeline (Chỉ load và clean 1 lần duy nhất để tiết kiệm RAM)
    raw_df = parser.load_data(file_path)
    clean_df = processor.clean_dataframe(raw_df)
    
    # 4. Aggregate cho cả 3 khung thời gian
    intervals = CONFIG['processing']['intervals']
    processed_package = {}
    
    for interval in intervals:
        logger.info(f"--- Đang vận hành Pipeline cho khung thời gian: {interval} ---")
        agg_df = processor.aggregate_data(clean_df, window=interval)
        
        # --- QUY TRÌNH LỌC NHIỄU DỮ LIỆU (DATA DENOISING) ---
        # Tự động loại bỏ các mẫu dữ liệu không đạt tiêu chuẩn (Nghi vấn Bot/DDoS)
        threshold = CONFIG.get('analysis', {}).get('bot_error_threshold', 0.8)
        clean_agg_df = agg_df[agg_df['error_rate'] < threshold].copy()
        
        # Xuất dữ liệu đã làm sạch ra file CSV phục vụ huấn luyện mô hình
        output_name = f"processed_{file_type}_{interval}.csv"
        clean_agg_df.to_csv(DATA_DIR / output_name, index=False)
        
        logger.info(f"Hoàn tất xuất tập dữ liệu sạch: {output_name} (Đã lọc {len(agg_df) - len(clean_agg_df)} mẫu nhiễu)")
        processed_package[interval] = clean_agg_df
        
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
