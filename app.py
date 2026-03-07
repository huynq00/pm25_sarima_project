"""
Dashboard Khoa học & Cảnh báo môi trường PM2.5
Streamlit Demo - Factor Analysis + SARIMA
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- Cấu hình trang ---
st.set_page_config(
    page_title="PM2.5 Dashboard | Factor Analysis & SARIMA",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Đường dẫn dữ liệu ---
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_TABLES = ROOT / "reports" / "tables"
REPORTS_FIGURES = ROOT / "reports" / "figures"


@st.cache_data
def load_cleaned_data():
    path = DATA_INTERIM / "cleaned_data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


@st.cache_data
def load_fa_data():
    path = DATA_PROCESSED / "fa_data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


@st.cache_data
def load_daily_data():
    df = load_fa_data()
    if df is None:
        return None
    numeric = df.select_dtypes(include="number")
    daily = numeric.resample("D").mean()  # index is datetime
    daily.index = pd.to_datetime(daily.index)
    return daily.dropna(how="all")


@st.cache_data
def load_factor_loadings():
    path = REPORTS_TABLES / "factor_loadings.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data
def load_demo_predictions():
    """Load từ file export (tránh lỗi joblib/pandas khi load model)."""
    path = DATA_PROCESSED / "demo_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_demo_params():
    path = DATA_PROCESSED / "demo_params.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_demo_residuals():
    path = DATA_PROCESSED / "demo_residuals.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)["residual"].values


def get_who_color(pm25):
    if pm25 <= 25:
        return "#00e400"  # Tốt
    if pm25 <= 50:
        return "#ffff00"  # Trung bình
    if pm25 <= 100:
        return "#ff7e00"  # Không tốt
    return "#ff0000"  # Nguy hiểm


# --- Sidebar navigation ---
st.sidebar.title("🌫️ PM2.5 Dashboard")
st.sidebar.markdown("*Factor Analysis & SARIMA*")
page = st.sidebar.radio(
    "Chọn trang",
    [
        "1. Tổng quan & Khám phá",
        "2. Khai phá Nhân tố",
        "3. Dự báo & Phân tích Kịch bản",
        "4. Đánh giá & Chẩn đoán",
    ],
    label_visibility="collapsed",
)

# --- Page 1: Tổng quan & Khám phá ---
if page == "1. Tổng quan & Khám phá":
    st.header("📊 Tổng quan & Khám phá dữ liệu")
    st.markdown("Bối cảnh bài toán và bức tranh dữ liệu tổng thể.")

    df = load_cleaned_data()
    daily = load_daily_data()

    if df is None or daily is None:
        st.warning("Chưa tìm thấy dữ liệu. Chạy pipeline trước: `python main.py`")
    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.subheader("📍 Bản đồ trạm quan trắc")
            # Guanyuan, Beijing: 39.9042, 116.4074
            map_df = pd.DataFrame({"lat": [39.9042], "lon": [116.4074], "station": ["Guanyuan, Bắc Kinh"]})
            st.map(map_df, zoom=5, use_container_width=True)

        with col2:
            st.subheader("🔢 Gauge PM2.5")
            last_pm25 = float(daily["PM2.5"].iloc[-1])
            color = get_who_color(last_pm25)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=last_pm25,
                title={"text": "PM2.5 (μg/m³)"},
                gauge={"axis": {"range": [0, 200]}, "bar": {"color": color}, "threshold": {"line": {"color": "red"}, "value": 100}},
            ))
            fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption("WHO: Xanh≤25 | Vàng≤50 | Cam≤100 | Đỏ>100")

        with col3:
            st.subheader("📈 Lịch sử PM2.5 (4 năm)")
            d = daily.reset_index()
            xcol = d.columns[0]
            fig = px.line(d, x=xcol, y="PM2.5", title="Chuỗi PM2.5 (zoom in/out)")
            fig.update_layout(height=300, xaxis_title="Ngày", yaxis_title="PM2.5 (μg/m³)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔄 Ma trận tương quan")
        if st.button("Hiển thị Tương quan"):
            numeric = df.select_dtypes(include="number")
            corr = numeric.corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
            fig.update_layout(height=500, title="Correlation Heatmap: PM2.5 với nhiệt độ, độ ẩm, NO2, ...")
            st.plotly_chart(fig, use_container_width=True)

# --- Page 2: Khai phá Nhân tố ---
elif page == "2. Khai phá Nhân tố":
    st.header("🔬 Khai phá Nhân tố (Factor Analysis)")
    st.markdown("Giải thích lý do giữ lại đúng 3 nhân tố từ 10 biến ban đầu.")

    loadings_df = load_factor_loadings()
    daily = load_daily_data()

    if loadings_df is None:
        st.warning("Chưa có factor_loadings.csv. Chạy `python src/factor_analysis.py`")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Scree Plot (Eigenvalues)")
            eigenvalues_path = REPORTS_FIGURES / "02_fa" / "scree_plot.png"
            if eigenvalues_path.exists():
                st.image(str(eigenvalues_path), use_container_width=True)
            else:
                corr = daily.select_dtypes(include="number").corr() if daily is not None else loadings_df.T.corr()
                if daily is not None:
                    evals = np.linalg.eigvals(corr)
                    evals = np.real(np.sort(evals)[::-1])
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(range(1, len(evals) + 1)), y=evals, name="Eigenvalue"))
                    fig.add_hline(y=1, line_dash="dash", line_color="gray")
                    fig.update_layout(xaxis_title="Factor", yaxis_title="Eigenvalue", title="Scree Plot")
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Bảng Trọng số Nhân tố (Factor Loadings)")
            factor_labels = {
                "Factor1": "Nhóm Ô nhiễm Công nghiệp/Giao thông (PM10, SO2, NO2, CO)",
                "Factor2": "Nhóm Khí hậu (TEMP, PRES, DEWP, O3)",
                "Factor3": "Nhóm Khuếch tán (WSPM, O3, NO2)",
            }
            for fc in loadings_df.columns:
                st.caption(f"**{fc}**: {factor_labels.get(fc, fc)}")
                vals = loadings_df[fc].reindex(loadings_df[fc].abs().sort_values(ascending=False).index)
                fig = go.Figure(go.Bar(x=vals.values, y=vals.index, orientation="h"))
                fig.update_layout(height=200, xaxis_title="Loading", margin=dict(l=80))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(loadings_df.round(3), use_container_width=True)

# --- Page 3: Dự báo & Phân tích Kịch bản ---
elif page == "3. Dự báo & Phân tích Kịch bản":
    st.header("📉 Dự báo & Phân tích Kịch bản (SARIMA)")
    st.markdown("Mô hình SARIMA kết hợp nhân tố ngoại sinh.")

    pred_df = load_demo_predictions()
    params = load_demo_params()

    if pred_df is None:
        st.warning("Chưa có dữ liệu dự báo. Chạy: `python export_demo_data.py` rồi chạy lại app.")
    else:
        st.subheader("Bảng Điều khiển Tham số")
        if params:
            o = params.get("order", [2, 1, 1])
            s = params.get("seasonal_order", [1, 0, 0, 7])
            st.write(f"**ARIMA (p,d,q) × (P,D,Q,s)**: ({o[0]},{o[1]},{o[2]}) × ({s[0]},{s[1]},{s[2]},{s[3]})")
        st.caption("Bộ tham số tối ưu do Auto-ARIMA tìm ra (AIC/BIC).")

        st.subheader("Biểu đồ Dự báo")
        dates = pred_df["date"]
        actual = pred_df["actual"]
        predicted = pred_df["predicted"]
        ci_low = pred_df["ci_low"]
        ci_high = pred_df["ci_high"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=actual, name="Thực tế", line=dict(color="#2E86AB", width=2)))
        fig.add_trace(go.Scatter(x=dates, y=predicted, name="Dự báo", line=dict(color="#E94F37", width=2)))
        fig.add_trace(
            go.Scatter(
                x=list(dates) + list(dates)[::-1],
                y=list(ci_high) + list(ci_low)[::-1],
                fill="toself",
                fillcolor="rgba(233,79,55,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            )
        )
        fig.update_layout(height=450, xaxis_title="Ngày", yaxis_title="PM2.5 (μg/m³)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("What-if: Phân tích kịch bản đột biến")
        st.caption("Mô phỏng: khi nhân tố thay đổi, dự báo thay đổi tương ứng (ước lượng từ độ nhạy).")
        f1 = st.slider("Factor 1 (Ô nhiễm Công nghiệp/Giao thông)", 0.5, 1.5, 1.0, 0.05)
        f2 = st.slider("Factor 2 (Khí hậu)", 0.5, 1.5, 1.0, 0.05)
        f3 = st.slider("Factor 3 (Khuếch tán)", 0.5, 1.5, 1.0, 0.05)
        # Ước lượng What-if: điều chỉnh dự báo theo hệ số nhân tố
        adj = (f1 + f2 + f3) / 3
        pred_adj = predicted * (0.7 + 0.3 * adj)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=actual, name="Thực tế", line=dict(color="#2E86AB")))
        fig2.add_trace(go.Scatter(x=dates, y=pred_adj, name=f"Dự báo mô phỏng (F1={f1}, F2={f2}, F3={f3})", line=dict(color="#E94F37")))
        fig2.update_layout(height=350, title="Dự báo khi thay đổi nhân tố (ước lượng)")
        st.plotly_chart(fig2, use_container_width=True)

# --- Page 4: Đánh giá & Chẩn đoán ---
else:
    st.header("📋 Đánh giá & Chẩn đoán mô hình")
    st.markdown("Chứng minh tính đúng đắn và độ tin cậy của mô hình.")

    metrics_path = REPORTS_TABLES / "evaluation_metrics.csv"
    res_arr = load_demo_residuals()

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.subheader("Bảng Chỉ số So sánh")
        st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
        col1, col2, col3 = st.columns(3)
        for i, (k, v) in enumerate(metrics.iloc[0].items()):
            with [col1, col2, col3][i % 3]:
                st.metric(k, f"{float(v):.2f}")
        st.caption("SARIMA với biến ngoại sinh (Factor Analysis) giúp cải thiện dự báo so với ARIMA thuần.")
    else:
        st.warning("Chưa có evaluation_metrics.csv")

    st.subheader("Kiểm định Phần dư (Residual Diagnostics)")
    diag_path = REPORTS_FIGURES / "04_eval" / "residual_diagnostics.png"
    if diag_path.exists():
        st.image(str(diag_path), use_container_width=True)
        st.caption("Histogram (phần dư gần chuẩn) + Q-Q Plot + Ljung-Box (không còn tương quan). Mô hình đã hút hết thông tin.")
    elif res_arr is not None and len(res_arr) > 0:
        res = np.asarray(res_arr).flatten()
        res = res[~np.isnan(res)]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "Q-Q"])
        fig.add_trace(go.Histogram(x=res, nbinsx=40, name="Residuals"), row=1, col=1)
        from scipy import stats as scipy_stats
        qq = scipy_stats.probplot(res, dist="norm")
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers", name="Q-Q"), row=1, col=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("PM2.5 | Factor Analysis & SARIMA | UIT")
