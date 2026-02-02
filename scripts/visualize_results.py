"""
SCRIPT: VISUALIZATION RUNNER
------------------------------------------------
Mô tả: Quét thư mục results, tìm file dự báo và gọi src để vẽ biểu đồ.
"""
import sys
import pandas as pd
from pathlib import Path

# --- SETUP IMPORT TỪ SRC ---
# Thêm thư mục gốc vào sys.path để import được src
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.visualize import plot_forecast_analysis

def main():
    # Cấu hình đường dẫn
    results_dir = root_dir / "results"
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách file kết quả (predictions_*.csv)
    files = list(results_dir.glob("predictions_*.csv"))
    
    if not files:
        print(f"❌ Không tìm thấy file kết quả nào trong: {results_dir}")
        print("👉 Hãy chạy 'evaluate.py' trước để có dữ liệu!")
        return

    print(f"\n{'='*60}")
    print(f"🎨 ĐANG VẼ BIỂU ĐỒ TỪ {len(files)} FILE KẾT QUẢ")
    print(f"{'='*60}")

    for file_path in files:
        # Lấy tên interval từ tên file (vd: predictions_15min.csv -> 15min)
        interval = file_path.stem.replace("predictions_", "")
        print(f"\n📂 Đang xử lý interval: {interval}...")
        
        try:
            # Đọc dữ liệu
            df = pd.read_csv(file_path)
            if 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])
            else:
                print(f"   ⚠️ File {file_path.name} thiếu cột 'ds', bỏ qua.")
                continue

            # GỌI HÀM VẼ TỪ SRC
            out_full, out_zoom = plot_forecast_analysis(df, interval, charts_dir)
            
            print(f"   ✅ Đã lưu Full: {out_full.name}")
            print(f"   ✅ Đã lưu Zoom: {out_zoom.name}")
            
        except Exception as e:
            print(f"   ❌ Lỗi khi vẽ {interval}: {e}")

    print(f"\n✨ HOÀN TẤT! Kiểm tra thư mục: {charts_dir}")

if __name__ == "__main__":
    main()