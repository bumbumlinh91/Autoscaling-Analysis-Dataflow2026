"""
MODULE: VISUALIZATION 
------------------------------------------------
Mô tả:
1. Đọc file results/predictions_{interval}.csv
2. Vẽ so sánh Actual vs (Prophet, XGBoost, LSTM).
3. Xuất ra 2 bản Zoom và Full.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# Cấu hình giao diện biểu đồ 
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 1.5

def visualize():
    results_dir = Path("results")
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách file kết quả
    files = list(results_dir.glob("predictions_*.csv"))
    if not files:
        print("❌ Không tìm thấy file kết quả nào trong thư mục 'results/'.")
        print("👉 Hãy chạy 'FinalFixerNoLog.py' trước!")
        return

    print(f"\n{'='*60}")
    print(f"🎨 ĐANG VẼ BIỂU ĐỒ ({len(files)} files)")
    print(f"{'='*60}")

    # Màu sắc định danh cho từng model
    colors = {
        'Actual': 'black',
        'Prophet': '#1f77b4',  # Xanh dương
        'XGBoost': '#ff7f0e',  # Cam
        'LSTM': '#d62728'      # Đỏ
    }

    for file_path in files:
        interval = file_path.stem.replace("predictions_", "")
        print(f"   >> Đang xử lý: {interval}")
        
        try:
            df = pd.read_csv(file_path)
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Xác định các cột model có trong file
            models = [c for c in df.columns if c not in ['ds', 'Actual']]
            
            # --- 1. BIỂU ĐỒ FULL (TOÀN CẢNH) ---
            fig, ax = plt.subplots()
            
            # Vẽ Actual (Mờ hơn chút để nổi bật model)
            ax.plot(df['ds'], df['Actual'], label='Thực tế (Actual)', 
                    color=colors['Actual'], alpha=0.3, linewidth=1)
            
            # Vẽ các Model
            for model in models:
                color = colors.get(model, 'blue') # Mặc định blue nếu ko có trong dict
                ax.plot(df['ds'], df[model], label=model, 
                        color=color, alpha=0.8, linewidth=1.2)
                
            ax.set_title(f"Dự báo Tải Server - Toàn cảnh ({interval})", fontweight='bold')
            ax.set_ylabel("Request Intensity")
            ax.set_xlabel("Thời gian")
            ax.legend(loc='upper right')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            out_full = charts_dir / f"chart_{interval}_full.png"
            plt.savefig(out_full, dpi=150)
            plt.close() # Giải phóng bộ nhớ
            
            # --- 2. BIỂU ĐỒ ZOOM (CẬN CẢNH 500 ĐIỂM ĐẦU) ---
            # Để thấy rõ chi tiết bám sát
            zoom_len = 500 if len(df) > 500 else len(df)
            df_zoom = df.head(zoom_len)
            
            fig, ax = plt.subplots()
            
            # Vẽ Actual (Đậm hơn ở chế độ zoom)
            ax.plot(df_zoom['ds'], df_zoom['Actual'], label='Thực tế', 
                    color=colors['Actual'], alpha=0.5, linewidth=2)
            
            for model in models:
                color = colors.get(model, 'blue')
                # Vẽ nét đứt cho model để dễ phân biệt với nền
                ax.plot(df_zoom['ds'], df_zoom[model], label=model, 
                        color=color, alpha=0.9, linewidth=2)

            ax.set_title(f"Chi tiết Dự báo - {interval} (Zoom {zoom_len} điểm đầu)", fontweight='bold')
            ax.set_ylabel("Request Intensity")
            ax.legend(loc='upper right')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
            plt.xticks(rotation=45)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            out_zoom = charts_dir / f"chart_{interval}_zoom.png"
            plt.savefig(out_zoom, dpi=150)
            plt.close()

        except Exception as e:
            print(f"   ⚠️ Lỗi khi vẽ {interval}: {e}")

    print(f"\n✅ XONG! Ảnh đã lưu tại thư mục: {charts_dir}")
    print("👉 Mở ảnh ra, copy vào báo cáo và đi ngủ đi!")

if __name__ == "__main__":
    visualize()