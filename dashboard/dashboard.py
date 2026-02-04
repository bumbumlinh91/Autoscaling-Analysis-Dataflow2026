import streamlit as st
import pandas as pd
import requests
import altair as alt
import numpy as np

def plot_comparison(df, title="So sánh Thực tế vs Dự báo"):
    # Biểu đồ Actual
    base = alt.Chart(df).encode(x=alt.X('ds:T', title='Thời gian', axis=alt.Axis(format='%H:%M')))
    
    line_actual = base.mark_line(color='gray', strokeDash=[5, 5], opacity=0.7).encode(
        y=alt.Y('y:Q', title='Requests/s'),
        tooltip=[alt.Tooltip('ds:T', format='%H:%M'), alt.Tooltip('y:Q', title='Thực tế')]
    )
    # Biểu đồ Forecast
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
st.sidebar.caption("Giá trị mặc định lấy từ Config")

# Lấy default từ config hoặc fallback
def_buffer = float(defaults.get("buffer_ratio", 0.2))
def_cooldown = int(defaults.get("cooldown_period", 3))

buffer_ratio = st.sidebar.slider("Hệ số dự phòng", 0.0, 1.0, def_buffer, 0.05)
cooldown_period = st.sidebar.slider("Thời gian hạ nhiệt", 0, 20, def_cooldown, 1)

tabs = st.tabs(["📊 1. Dự báo & Thực tế", "⚖️ 2. Kế hoạch Autoscaling", "💰 3. Phân tích chi phí"])

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
    st.subheader("AI dự báo chính xác đến đâu?")
    st.markdown("So sánh tải thực tế và tải dự báo để đánh giá độ tin cậy của mô hình.")
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
            st.warning("⚠️ Không có dữ liệu thực tế trong file kết quả để so sánh.")
            st.altair_chart(plot_interactive(df, "yhat", "#00CC96", "Dự báo tải", "Requests/s"), use_container_width=True)
        
        with st.expander("Xem dữ liệu chi tiết"):
            st.dataframe(
                df.rename(columns={"ds": "Thời gian", "y": "Thực tế", "yhat": "Dự báo"}), 
                use_container_width=True
            )

with tabs[1]:
    st.subheader("Hệ thống phản ứng thế nào?")
    st.markdown("Dựa trên dự báo, hệ thống đề xuất số lượng Server cần thiết để đảm bảo SLA.")
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
            base = alt.Chart(df).encode(x=alt.X('ds:T', title='Thời gian', axis=alt.Axis(format='%H:%M')))
            line_load = base.mark_line(color='#00CC96').encode(y=alt.Y('yhat:Q', title='Tải dự báo'), tooltip=['ds', 'yhat'])
            line_rep = base.mark_line(interpolate='step-after', color='#FF4B4B').encode(y=alt.Y('recommended_replicas:Q', title='Số Replicas'), tooltip=['ds', 'recommended_replicas'])
            
            st.altair_chart((line_load + line_rep).resolve_scale(y='independent').properties(title="Tải dự báo vs Số Replicas đề xuất"), use_container_width=True)
            
        with c2:
            st.write("📋 **Nhật ký hành động**")
            st.dataframe(
                df[["ds", "recommended_replicas", "action", "reason"]].rename(columns={
                    "ds": "Thời gian",
                    "recommended_replicas": "Replicas đề xuất",
                    "action": "Hành động",
                    "reason": "Lý do"
                }),
                use_container_width=True
            )

with tabs[2]:
    st.subheader("Phân tích chi phí và lợi ích")
    st.markdown("So sánh chi phí giữa việc dùng **AI Autoscaling** và **Reactive Scaling**.")
    
    # Lấy default cost từ config
    def_cost = float(defaults.get("server_cost", 0.05))      
    def_cap = float(defaults.get("server_capacity", 5000000)) 
    def_penalty = float(defaults.get("sla_penalty", 0.0001)) 

    # Input giả định chi phí
    c_cost1, c_cost2, c_cost3 = st.columns(3)
    with c_cost1:
        server_cost = st.number_input("Chi phí Server ($/giờ/replica)", value=def_cost, step=0.1)
    with c_cost2:
        server_capacity = st.number_input("Sức chịu tải (Req/replica)", value=def_cap, step=10000.0)
    with c_cost3:
        sla_penalty = st.number_input("Phạt SLA ($/req rớt)", value=def_penalty, step=0.0001, format="%.4f")

    # Lấy dữ liệu từ API
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
        
        # --- BƯỚC 1: CHUẨN BỊ DỮ LIỆU ---
        if "y" not in df.columns or df["y"].isna().all():
            st.error("⚠️ Dữ liệu thực tế (y) bị thiếu trong API response! Vui lòng chạy lại 'evaluate.py' để cập nhật kết quả.")
            df["y"] = 0
            
        df["y"] = df["y"].fillna(0) # Đảm bảo không còn NaN

        # --- BƯỚC 2: MÔ PHỎNG REACTIVE  ---
        # Reactive bị trễ 1 nhịp (Lag) so với thực tế
        # Shift(1) lấy giá trị của interval trước đó
        lagged_load = df["y"].shift(1).fillna(df["yhat"])
        
        # Buffer 20% 
        df["reactive_replicas"] = np.ceil((lagged_load / server_capacity) * 1.2).astype(int)
        df["reactive_replicas"] = df["reactive_replicas"].clip(lower=1) 
        
        # --- BƯỚC 3: TÍNH TOÁN CHI PHÍ ---
        hours_per_point = 15 / 60 
        if "1min" in interval: hours_per_point = 1/60
        elif "5min" in interval: hours_per_point = 5/60
            
        # 1. Chi phí Hạ tầng
        df["infra_ai"] = df["recommended_replicas"] * server_cost * hours_per_point
        df["infra_reactive"] = df["reactive_replicas"] * server_cost * hours_per_point
        
        # 2. Chi phí Phạt SLA
        # Capacity thực tế
        cap_ai = df["recommended_replicas"] * server_capacity
        cap_reactive = df["reactive_replicas"] * server_capacity
        
        # Request bị rớt (Chỉ tính khi Nhu cầu > Khả năng)
        df["dropped_ai"] = (df["y"] - cap_ai).clip(lower=0)
        df["dropped_reactive"] = (df["y"] - cap_reactive).clip(lower=0)
        
        df["penalty_ai"] = df["dropped_ai"] * sla_penalty
        df["penalty_reactive"] = df["dropped_reactive"] * sla_penalty
            
        # 3. Tổng kết
        df["total_ai"] = df["infra_ai"] + df["penalty_ai"]
        df["total_reactive"] = df["infra_reactive"] + df["penalty_reactive"]
        
        total_ai = df["total_ai"].sum()
        total_reactive = df["total_reactive"].sum()
        savings = total_reactive - total_ai
        roi = (savings / total_reactive * 100) if total_reactive > 0 else 0
        # --- SLA: % request phục vụ thành công ---
        total_requests = df["y"].sum()

        sla_ai = 1 - (df["dropped_ai"].sum() / total_requests) if total_requests > 0 else 1
        sla_reactive = 1 - (df["dropped_reactive"].sum() / total_requests) if total_requests > 0 else 1

        sla_ai_pct = sla_ai * 100
        sla_reactive_pct = sla_reactive * 100
        # --- HIỂN THỊ METRICS ---
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Tổng chi phí (Reactive)", f"${total_reactive:,.2f}")
        m2.metric("Tổng chi phí (AI Model)", f"${total_ai:,.2f}", delta_color="inverse")
        
        delta_val = f"{savings:,.2f}"
        if savings > 0: delta_val = f"+{delta_val}"
        
        m3.metric("Tiết kiệm (Savings)", f"${savings:,.2f}", delta=delta_val)
        m4.metric("ROI (%)", f"{roi:.2f}%")
        m5.metric(
                    "SLA (%)",
                    f"{sla_ai_pct:.2f}%",
                    delta=f"{(sla_ai_pct - sla_reactive_pct):+.2f}%"
                )

        # --- BIỂU ĐỒ ---
        st.markdown("#### 📉 So sánh Quy mô Server")
        chart_data = df.melt(id_vars=["ds"], value_vars=["recommended_replicas", "reactive_replicas"], 
                             var_name="Chiến lược Scaling", value_name="Replicas")
        chart_data["Chiến lược Scaling"] = chart_data["Chiến lược Scaling"].map({
            "recommended_replicas": "AI Dự báo (Predictive)",
            "reactive_replicas": "Truyền thống (Reactive)"
        })

        c = alt.Chart(chart_data).mark_line(interpolate='step-after').encode(
            x=alt.X('ds:T', title='Thời gian', axis=alt.Axis(format='%H:%M')),
            y=alt.Y('Replicas:Q', title='Số lượng Server'),
            color=alt.Color('Chiến lược Scaling', scale=alt.Scale(range=['#00CC96', '#FF4B4B'])),
            tooltip=[
                alt.Tooltip('ds:T', title='Thời gian', format='%H:%M'),
                alt.Tooltip('Chiến lược Scaling'),
                alt.Tooltip('Replicas')
            ]
        ).interactive()
        st.altair_chart(c, use_container_width=True)

        # Biểu đồ rớt request để chứng minh tại sao Reactive phạt nặng
        total_dropped_reactive = df["dropped_reactive"].sum()
        total_dropped_ai = df["dropped_ai"].sum()

        if total_dropped_reactive > 0 or total_dropped_ai > 0:
            st.markdown("#### 📉 Phân tích Request bị rớt (Nguyên nhân mất tiền SLA)")
            st.caption(f"Biểu đồ dưới đây so sánh lượng request bị rớt giữa hai chiến lược. "
                       f"Reactive thường bị rớt do độ trễ khi scale up, dẫn đến phạt SLA cao. "
                       f"(Reactive: {int(total_dropped_reactive):,} vs AI: {int(total_dropped_ai):,})")
            
            drop_data = df.melt(
                id_vars=["ds"], 
                value_vars=["dropped_reactive", "dropped_ai"], 
                var_name="Strategy", 
                value_name="Dropped"
            )
            
            drop_data["Strategy"] = drop_data["Strategy"].map({
                "dropped_reactive": "Reactive (Truyền thống)",
                "dropped_ai": "AI Model (Predictive)"
            })

            c_drop = alt.Chart(drop_data).mark_area(opacity=0.6).encode(
                x=alt.X('ds:T', title='Thời gian', axis=alt.Axis(format='%H:%M')),
                y=alt.Y('Dropped:Q', title="Số lượng Request bị rớt"),
                color=alt.Color('Strategy', scale=alt.Scale(domain=['Reactive (Truyền thống)', 'AI Model (Predictive)'], range=['#FF4B4B', '#00CC96'])),
                tooltip=[
                    alt.Tooltip('ds:T', format='%H:%M'),
                    alt.Tooltip('Strategy', title='Chiến lược'),
                    alt.Tooltip('Dropped:Q', format=',.0f', title='Request rớt')
                ]
            ).properties(height=250).interactive()
            
            st.altair_chart(c_drop, use_container_width=True)