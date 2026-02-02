import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Dataflow 2026 Autoscaling Dashboard", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")

st.title("📈 Dataflow 2026 - Forecast & Autoscaling Dashboard")

interval = st.sidebar.selectbox("Interval", ["1min", "5min", "15min"], index=2)
model = st.sidebar.selectbox("Model", ["xgboost", "prophet", "lstm"], index=0)
target = st.sidebar.selectbox("Target", ["intensity", "num_requests", "total_bytes"], index=0)

horizon = st.sidebar.number_input(
    "Horizon (steps)", min_value=1, value=96, step=12,
    help="Số bước dự báo (>0). Ví dụ: 1min=60, 5min=288, 15min=96"
)

st.sidebar.markdown("### Scaling Policy")
buffer_ratio = st.sidebar.slider("buffer_ratio", 0.0, 1.0, 0.2, 0.05)
cooldown_period = st.sidebar.slider("cooldown_period", 0, 20, 3, 1)

tabs = st.tabs(["Overview", "Forecast", "Autoscaling Plan"])

def post_json(path, payload):
    url = f"{API_BASE}{path}"
    st.caption(f"Calling: {url}")
    try:
        r = requests.post(url, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Cannot reach API at {API_BASE}. Make sure FastAPI is running.\n\n{e}")
        return None

    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    return r.json()

def forecast_payload():
    return {
        "interval": interval,
        "model": model,
        "horizon": int(horizon),
        "target": target,
    }

with tabs[0]:
    st.subheader("Overview")
    data = post_json("/forecast", forecast_payload())
    if data:
        df = pd.DataFrame(data["points"])
        df["ds"] = pd.to_datetime(df["ds"])
        st.metric("Rows", len(df))
        st.line_chart(df.set_index("ds")[["yhat"]])

with tabs[1]:
    st.subheader("Forecast")
    data = post_json("/forecast", forecast_payload())
    if data:
        df = pd.DataFrame(data["points"])
        df["ds"] = pd.to_datetime(df["ds"])
        st.dataframe(df.head(50), use_container_width=True)
        st.line_chart(df.set_index("ds")[["yhat"]])

with tabs[2]:
    st.subheader("Autoscaling Recommendation")
    payload = {
        "interval": interval,
        "model": model,
        "horizon": int(horizon),
        "target": target,
        "policy_params": {
            "buffer_ratio": float(buffer_ratio),
            "cooldown_period": int(cooldown_period),
        }
    }

    data = post_json("/recommend-scaling", payload)
    if data:
        df = pd.DataFrame(data["points"])
        df["ds"] = pd.to_datetime(df["ds"])

        st.caption("Policy used:")
        st.json(data["policy_used"])

        c1, c2 = st.columns([2, 1])
        with c1:
            st.line_chart(df.set_index("ds")[["yhat", "recommended_replicas"]])
        with c2:
            st.write("Scale actions")
            st.dataframe(
                df[["ds", "recommended_replicas", "action", "reason"]].head(200),
                use_container_width=True
            )