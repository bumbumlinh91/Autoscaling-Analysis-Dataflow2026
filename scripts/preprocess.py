"""
SCRIPT: DATA PREPROCESSING RUNNER
------------------------------------------------
Mô tả: Gọi logic từ src/data_preprocessing.py để làm sạch dữ liệu raw.
"""
import sys
from pathlib import Path

# 1. SETUP ĐƯỜNG DẪN (Để tìm thấy src)
# Lấy thư mục gốc 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# 2. IMPORT TỪ SRC
from src.data_preprocessing import run_full_pipeline

def main():
    print(f"\n{'='*60}")
    print("🧹 BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU (PRE-PROCESSING)")
    print(f"{'='*60}")
    
    # Chạy cho cả tập train và test
    data_types = ['train', 'test']
    
    try:
        for dtype in data_types:
            print(f"\n>>> ĐANG XỬ LÝ TẬP: {dtype.upper()}")
            # Gọi hàm từ src
            run_full_pipeline(file_type=dtype)
            
        print(f"\n{'='*60}")
        print("✅ TIỀN XỬ LÝ HOÀN TẤT!")
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()