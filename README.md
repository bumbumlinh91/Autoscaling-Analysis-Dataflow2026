# Autoscaling-Analysis-Dataflow2026
Hệ thống dự báo tải và tự động điều chỉnh máy chủ nhằm tối ưu chi phí vận hành
---

## 1. Tóm tắt
Dự án tập trung vào **dự báo lưu lượng web** từ HTTP access logs và sử dụng kết quả dự báo để thiết kế **predictive autoscaling**, nhằm giảm chi phí hạ tầng và hạn chế dropped requests so với reactive autoscaling truyền thống.

Cách tiếp cận:
- Dự báo traffic ở các khung 1m / 5m / 15m
- So sánh Prophet, XGBoost, LSTM
- Dùng forecast làm tín hiệu điều khiển autoscaling

---

## 2. Dữ liệu
- **Nguồn**: NASA HTTP Web Server Logs (07–08/1995)
- **Trường chính**: timestamp, request, status_code, bytes
- **Tiền xử lý**:
  - Parse & chuẩn hóa thời gian
  - Loại bỏ server outage (01/08 → 03/08/1995)
  - Tổng hợp theo 1m / 5m / 15m
  - Feature engineering: lags (24h, 7d), rolling stats, calendar features

---

## 3. Mô hình & Kiến trúc
- **Baseline**: Naive, Seasonal Naive  
- **Mô hình**: Prophet, XGBoost, LSTM  
- **Training**:
  - Time-based split (train trước, test sau)
  - Rolling validation
- **Chống data leakage**:
  - Không shuffle
  - Chỉ dùng dữ liệu quá khứ cho lags/rolling

Pipeline:
Logs → Preprocess → Aggregate → Forecast → Autoscaling → Cost & SLA Eval


---

## 4. Đánh giá
- **Metrics**: RMSE, MAE, MSE, MAPE
- **Nhận xét chính**:
  - 1m: nhiễu cao, khó bắt spike
  - 5m: cân bằng tốt nhất cho autoscaling
  - 15m: ổn định, phù hợp planning
- **So sánh mô hình**:
  - Prophet: mượt, bỏ lỡ spike
  - XGBoost: bám tốt biến động ngắn hạn
  - LSTM: ổn định nhất ở 15m nhưng chi phí cao

---

## 5. Triển khai

```bash
# Tạo môi trường
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
# Huấn luyện
python scripts/preprocess.py && \
python scripts/feature_eng.py && \
python scripts/train.py && \
python scripts/evaluate.py && \
python scripts/simulate.py && \
python scripts/visualize_results.py
# Chạy API, dashboard
python scripts/run_api.py
python scripts/run_dashboard.py
```
- API endpoints:
  - `POST /forecast`
  - `POST /recommend-scaling`
- Demo UI: Streamlit dashboard hiển thị kết quả dự báo và quyết định autoscaling

---

## 6. Kết quả & Ứng dụng

- **Giảm 34.14% chi phí vận hành**
- **Giảm ~110 triệu dropped requests**
- SLA error rate giảm từ **3.10% → 2.00%**

Ứng dụng cho web services, API gateways và các hệ thống autoscaling trên nền tảng cloud.

---

## 7. Giới hạn & Hướng phát triển

- Spike bất thường chưa được mô hình hóa tường minh
- Chưa liên kết trực tiếp forecast với latency
- Hướng mở rộng:
  - Anomaly detection
  - Ensemble forecasting
  - Probabilistic forecasting
  - Transformer-based models

---

## 8. Tác giả & License

- **Tác giả**: *FunnyGuys*
- **License**: MIT