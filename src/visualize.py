"""
MODULE: VISUALIZATION LIBRARY 
------------------------------------------------
Mô tả: Chứa logic cốt lõi để vẽ biểu đồ so sánh Model và biểu đồ Zoom.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# Cấu hình giao diện chung
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 1.5

def plot_forecast_analysis(df, interval, output_dir):
    """
    Vẽ và lưu 2 biểu đồ: Toàn cảnh (Full) và Cận cảnh (Zoom).
    
    Args:
        df (pd.DataFrame): Dữ liệu chứa cột 'ds', 'Actual' và các model.
        interval (str): Tên khoảng thời gian (15min, 5min...).
        output_dir (Path): Thư mục lưu ảnh.
    """
    output_dir = Path(output_dir)
    
    # Xác định các model có trong dữ liệu (trừ ds và Actual)
    models = [c for c in df.columns if c not in ['ds', 'Actual']]
    
    # Bảng màu cố định cho đồng bộ
    colors = {
        'Actual': 'black',
        'Prophet': '#1f77b4',  # Xanh dương
        'XGBoost': '#ff7f0e',  # Cam
        'LSTM': '#2ca02c'      # Xanh lá
    }

    # --- 1. BIỂU ĐỒ TOÀN CẢNH (FULL) ---
    fig_full, ax = plt.subplots()
    
    # Vẽ Actual
    ax.plot(df['ds'], df['Actual'], label='Thực tế', 
            color=colors['Actual'], alpha=0.3, linewidth=1)
    
    # Vẽ Models
    for model in models:
        color = colors.get(model, 'blue') # Mặc định blue nếu model lạ
        ax.plot(df['ds'], df[model], label=model, 
                color=color, alpha=0.8, linewidth=1.5)

    ax.set_title(f"So sánh Hiệu năng Mô hình - {interval} (Toàn cảnh)", fontweight='bold')
    ax.set_ylabel("Request Intensity")
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    # Lưu Full
    out_full = output_dir / f"chart_full_{interval}.png"
    plt.tight_layout()
    plt.savefig(out_full, dpi=150)
    plt.close(fig_full) # Đóng để giải phóng RAM

    # --- 2. BIỂU ĐỒ ZOOM (500 ĐIỂM ĐẦU) ---
    zoom_len = 500 if len(df) > 500 else len(df)
    df_zoom = df.head(zoom_len)
    
    fig_zoom, ax = plt.subplots()
    
    # Vẽ Actual (Đậm hơn chút ở chế độ zoom)
    ax.plot(df_zoom['ds'], df_zoom['Actual'], label='Thực tế', 
            color=colors['Actual'], alpha=0.5, linewidth=2)
    
    for model in models:
        color = colors.get(model, 'blue')
        # Vẽ nét đứt hoặc alpha cao để dễ nhìn
        ax.plot(df_zoom['ds'], df_zoom[model], label=model, 
                color=color, alpha=0.9, linewidth=2)

    ax.set_title(f"Chi tiết Dự báo - {interval} (Zoom {zoom_len} điểm đầu)", fontweight='bold')
    ax.set_ylabel("Request Intensity")
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Lưu Zoom
    out_zoom = output_dir / f"chart_zoom_{interval}.png"
    plt.tight_layout()
    plt.savefig(out_zoom, dpi=150)
    plt.close(fig_zoom)

    return out_full, out_zoom