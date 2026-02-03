import streamlit as st
import pandas as pd
import requests
import altair as alt
import numpy as np

def plot_comparison(df, title="So sánh Thực tế vs Dự báo"):
    # Biểu đồ Actual (Thực tế)
    base = alt.Chart(df).encode(x=alt.X('ds:T', title='Thời gian', axis=alt.Axis(format='%H:%M')))
    
    line_actual = base.mark_line(color='gray', strokeDash=[5, 5], opacity=0.7).encode(
        y=alt.Y('y:Q', title='Requests/s'),
        tooltip=[alt.Tooltip('ds:T', format='%H:%M'), alt.Tooltip('y:Q', title='Thực tế')]
    )
    # Biểu đồ Forecast (Dự báo)
    line_forecast = base.mark_line(color='#00CC96').encode(
        y=alt.Y('yhat:Q'),
        tooltip=[alt.Tooltip('ds:T', format='%H:%M'), alt.Tooltip('yhat:Q', title='Dự báo')]
    )
    
    chart = (line_actual + line_forecast).properties(title=title, height=400).interactive()
    return chart

def plot_interactive(df, y_col, color_hex="#FF4B4B", title="Biểu đồ", y_label="Giá trị"):
    chart = alt.Chart(df).mark_line(color=color_hex).encode(
        x=alt.X('ds:T', title='Thời gian', axis=alt.Axis(format='%d/%m %H:%M')),
        y=alt.Y(f'{y_col}:Q', title=y_label),
        tooltip=[
            alt.Tooltip('ds:T', title='Thời gian', format='%Y-%m-%d %H:%M'),
            alt.Tooltip(f'{y_col}:Q', title=y_label, format=',.2f')
        ]
    ).properties(title=title, height=400).interactive()
    return chart
st.set_page_config(page_title="Dataflow 2026 Autoscaling Dashboard", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")

# --- LOAD CONFIG TỪ API ---
@st.cache_data
def get_default_config(api_url):
    try:
        resp = requests.get(f"{api_url}/config", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return {}

defaults = get_default_config(API_BASE)

st.title("📈 Dataflow 2026 - Hệ thống Dự báo & Autoscaling")

interval = st.sidebar.selectbox("Interval", ["1min", "5min", "15min"], index=2)
model = st.sidebar.selectbox("Model", ["XGBoost", "Prophet", "LSTM"], index=0)

horizon = st.sidebar.number_input(
    "Horizon (số bước dự báo)", min_value=1, value=96, step=12,
    help="Số bước dự báo (>0). Ví dụ: 1min=60, 5min=288, 15min=96"
)

st.sidebar.markdown("### ⚙️ Chính sách Scaling")
st.sidebar.caption("Giá trị mặc định được tải từ config.yaml")

# Lấy default từ config hoặc fallback
def_buffer = float(defaults.get("buffer_ratio", 0.2))
def_cooldown = int(defaults.get("cooldown_period", 3))

buffer_ratio = st.sidebar.slider("Hệ số dự phòng (Buffer Ratio)", 0.0, 1.0, def_buffer, 0.05)
cooldown_period = st.sidebar.slider("Thời gian hạ nhiệt (Cooldown)", 0, 20, def_cooldown, 1)

tabs = st.tabs(["📊 1. Dự báo & Thực tế", "⚖️ 2. Kế hoạch Autoscaling", "💰 3. Bài toán Tài chính"])

def post_json(path, payload):
    url = f"{API_BASE}{path}"
    # st.caption(f"Calling: {url}")
    try:
        r = requests.post(url, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Không thể kết nối API tại {API_BASE}. Hãy chắc chắn API đang chạy.\n\n{e}")
        return None

    if r.status_code != 200:
        st.error(f"Lỗi API {r.status_code}: {r.text}")
        return None
    return r.json()

def forecast_payload():
    return {
        "interval": interval,
        "model": model.lower(),
        "horizon": int(horizon),
        "target": "intensity",  # Mặc định intensity
    }

with tabs[0]:
    st.subheader("Câu chuyện 1: AI dự báo chính xác đến đâu?")
    st.markdown("So sánh tải thực tế (Actual) và tải dự báo (Forecast) để đánh giá độ tin cậy của mô hình.")
    data = post_json("/forecast", forecast_payload())
    if data:
        df = pd.DataFrame(data["points"])
        df["ds"] = pd.to_datetime(df["ds"])
        c1, c2 = st.columns(2)
        c1.metric("Số điểm dữ liệu", len(df))
        if "y" in df.columns and df["y"].notna().any():
            mae = np.mean(np.abs(df["y"] - df["yhat"]))
            c2.metric("Sai số trung bình (MAE)", f"{mae:.2f}")
            st.altair_chart(plot_comparison(df), use_container_width=True)
        else:
            st.warning("⚠️ Không có dữ liệu thực tế (Actual) trong file kết quả để so sánh.")
            st.altair_chart(plot_interactive(df, "yhat", "#00CC96", "Dự báo tải", "Requests/s"), use_container_width=True)
        
        with st.expander("Xem dữ liệu chi tiết"):
            st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.subheader("Câu chuyện 2: Hệ thống phản ứng thế nào?")
    st.markdown("Dựa trên dự báo, hệ thống đề xuất số lượng Server (Replicas) cần thiết để đảm bảo SLA.")
    payload = {
        "interval": interval,
        "model": model.lower(),
        "horizon": int(horizon),
        "target": "intensity",
        "policy_params": {
            "buffer_ratio": float(buffer_ratio),
            "cooldown_period": int(cooldown_period),
        }
    }

    data = post_json("/recommend-scaling", payload)
    if data:
        df = pd.DataFrame(data["points"])
        df["ds"] = pd.to_datetime(df["ds"])


        c1, c2 = st.columns([2, 1])
        with c1:
            # Vẽ biểu đồ Replicas vs Load
            base = alt.Chart(df).encode(x=alt.X('ds:T', axis=alt.Axis(format='%H:%M')))
            line_load = base.mark_line(color='#00CC96').encode(y=alt.Y('yhat:Q', title='Tải dự báo'), tooltip=['ds', 'yhat'])
            line_rep = base.mark_line(interpolate='step-after', color='#FF4B4B').encode(y=alt.Y('recommended_replicas:Q', title='Số Replicas'), tooltip=['ds', 'recommended_replicas'])
            
            st.altair_chart((line_load + line_rep).resolve_scale(y='independent').properties(title="Tải dự báo vs Số Replicas đề xuất"), use_container_width=True)
            
        with c2:
            st.write("📋 **Nhật ký hành động (Action Log)**")
            st.dataframe(
                df[["ds", "recommended_replicas", "action", "reason"]],
                use_container_width=True
            )

with tabs[2]:
    st.subheader("Câu chuyện 3: Bài toán kinh tế (ROI)")
    st.markdown("So sánh chi phí giữa việc dùng **AI Autoscaling** và **Reactive Scaling (Truyền thống)**.")
    
    # Lấy default cost từ config
    def_cost = float(defaults.get("server_cost", 0.5))
    def_cap = float(defaults.get("server_capacity", 500000))
    def_penalty = float(defaults.get("sla_penalty", 0.001))

    
    # Input giả định chi phí
    c_cost1, c_cost2, c_cost3 = st.columns(3)
    with c_cost1:
        server_cost = st.number_input("Chi phí Server ($/giờ/replica)", value=def_cost, step=0.1)
    with c_cost2:
        server_capacity = st.number_input("Sức chịu tải (Req/replica)", value=def_cap, step=10000.0)
    with c_cost3:
        sla_penalty = st.number_input("Phạt SLA ($/req rớt)", value=def_penalty, step=0.0001, format="%.4f")

    # Lấy dữ liệu từ API (sử dụng lại payload cũ)
    payload = {
        "interval": interval,
        "model": model.lower(),
        "horizon": int(horizon),
        "target": "intensity",
        "policy_params": {
            "buffer_ratio": float(buffer_ratio),
            "cooldown_period": int(cooldown_period),
        }
    }
    data = post_json("/recommend-scaling", payload)
    
    if data:
        df = pd.DataFrame(data["points"])
        df["ds"] = pd.to_datetime(df["ds"])
        
        # --- MÔ PHỎNG REACTIVE (BASELINE) ---
        # Reactive thuần túy: Scale theo nhu cầu thực tế (ở đây là forecast) / capacity
        # Thường Reactive sẽ scale dư ra một chút để an toàn (ví dụ +10%)
        lagged_load = df["y"].shift(1).fillna(df["yhat"])
        df["reactive_replicas"] = np.ceil((lagged_load / server_capacity) * 1.2).astype(int)
        df["reactive_replicas"] = df["reactive_replicas"].clip(lower=1) # [FIX] Tối thiểu 1 server
        
        # --- TÍNH TOÁN CHI PHÍ & SLA PENALTY ---
        # Giả sử interval là 15min -> mỗi điểm dữ liệu tốn: server_cost * (15/60)
        hours_per_point = 15 / 60  # Mặc định logic 15p, nếu interval khác cần chỉnh
        if "1min" in interval: hours_per_point = 1/60
        elif "5min" in interval: hours_per_point = 5/60
            
        # 1. Chi phí Hạ tầng (Infrastructure Cost)
        df["infra_ai"] = df["recommended_replicas"] * server_cost * hours_per_point
        df["infra_reactive"] = df["reactive_replicas"] * server_cost * hours_per_point
        
        # 2. Chi phí Phạt SLA (Penalty Cost)
        # Nếu có dữ liệu thực tế (y), tính số request bị rớt
        if "y" in df.columns and df["y"].notna().any():
            actual_load = df["y"].fillna(0)
            # Capacity thực tế của hệ thống
            cap_ai = df["recommended_replicas"] * server_capacity
            cap_reactive = df["reactive_replicas"] * server_capacity
            
            # Số request bị rớt (Dropped) = Nhu cầu - Khả năng đáp ứng
            df["dropped_ai"] = (actual_load - cap_ai).clip(lower=0)
            df["dropped_reactive"] = (actual_load - cap_reactive).clip(lower=0)
            
            df["penalty_ai"] = df["dropped_ai"] * sla_penalty
            df["penalty_reactive"] = df["dropped_reactive"] * sla_penalty
        else:
            df["penalty_ai"] = 0.0
            df["penalty_reactive"] = 0.0
            
        # 3. Tổng chi phí
        df["total_ai"] = df["infra_ai"] + df["penalty_ai"]
        df["total_reactive"] = df["infra_reactive"] + df["penalty_reactive"]
        
        total_ai = df["total_ai"].sum()
        total_reactive = df["total_reactive"].sum()
        savings = total_reactive - total_ai
        roi = (savings / total_reactive * 100) if total_reactive > 0 else 0
        
        # --- HIỂN THỊ METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng chi phí (Reactive)", f"${total_reactive:,.2f}")
        m2.metric("Tổng chi phí (AI Model)", f"${total_ai:,.2f}", delta_color="inverse")
        
        # Hiển thị màu sắc đúng logic: Dương là tốt (Xanh), Âm là lỗ (Đỏ)
        delta_val = f"{savings:,.2f}"
        if savings > 0: delta_val = f"+{delta_val}"
        
        m3.metric("Tiết kiệm (Savings)", f"${savings:,.2f}", delta=delta_val)
        m4.metric("ROI (%)", f"{roi:.2f}%")
        
        # --- BIỂU ĐỒ SO SÁNH ---
        st.markdown("#### 📉 So sánh chiến lược Scaling")
        chart_data = df.melt(id_vars=["ds"], value_vars=["recommended_replicas", "reactive_replicas"], 
                             var_name="Strategy", value_name="Replicas")
        
        c = alt.Chart(chart_data).mark_line().encode(
            x=alt.X('ds:T', title='Thời gian', axis=alt.Axis(format='%H:%M')),
            y=alt.Y('Replicas:Q'),
            color=alt.Color('Strategy', legend=alt.Legend(title="Chiến lược"), 
                            scale=alt.Scale(domain=['recommended_replicas', 'reactive_replicas'], range=['#00CC96', '#FF4B4B'])),
            tooltip=['ds', 'Strategy', 'Replicas']
        ).interactive()
        st.altair_chart(c, use_container_width=True)
