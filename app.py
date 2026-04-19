"""
Dashboard Khoa học & Cảnh báo môi trường PM2.5
Streamlit Demo - Factor Analysis + SARIMAX
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Cấu hình trang ---
st.set_page_config(
    page_title="PM2.5 Dashboard | Factor Analysis & SARIMAX",
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


@st.cache_data
def load_demo_models_comparison():
    """Wide CSV: actual + SARIMAX / RF / XGB / LSTM / hybrid (từ export_demo_data)."""
    path = DATA_PROCESSED / "demo_models_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_demo_models_metrics():
    path = DATA_PROCESSED / "demo_models_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE / MAE / MAPE giữa thực tế và một chuỗi dự báo (cùng độ dài)."""
    y = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan")}

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    mape = float(np.mean(np.abs((y - p) / (y + 1e-10))) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


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
st.sidebar.markdown("*Factor Analysis & SARIMAX*")
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

            daily_reset = daily.reset_index()
            date_col = daily_reset.columns[0]
            available_dates = pd.to_datetime(daily_reset[date_col]).dt.date.tolist()

            if "pm25_selected_date" not in st.session_state:
                st.session_state.pm25_selected_date = available_dates[-1]

            c1, c2 = st.columns(2)

            with c1:
                if st.button("Ngày mới nhất", use_container_width=True):
                    st.session_state.pm25_selected_date = available_dates[-1]

            with c2:
                if st.button("Ngày trước đó", use_container_width=True):
                    current_date = st.session_state.pm25_selected_date
                    if current_date in available_dates:
                        current_idx = available_dates.index(current_date)
                        if current_idx > 0:
                            st.session_state.pm25_selected_date = available_dates[current_idx - 1]

            selected_date = st.date_input(
                "Chọn ngày",
                min_value=available_dates[0],
                max_value=available_dates[-1],
                key="pm25_selected_date",
            )

            selected_row = daily_reset[
                pd.to_datetime(daily_reset[date_col]).dt.date == selected_date
            ]

            if selected_row.empty:
                st.warning("Không có dữ liệu cho ngày đã chọn.")
            else:
                selected_pm25 = float(selected_row["PM2.5"].iloc[0])
                color = get_who_color(selected_pm25)

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=selected_pm25,
                    title={"text": f"PM2.5 (μg/m³) - {selected_date.strftime('%d/%m/%Y')}"},
                    gauge={
                        "axis": {"range": [0, 200]},
                        "bar": {"color": color},
                        "threshold": {
                            "line": {"color": "red"},
                            "value": 100
                        },
                    },
                ))
                fig_g.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
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
    st.header("📉 Dự báo & Phân tích kịch bản")
    st.markdown(
        "So sánh nhiều mô hình trên **cùng tập test** (20% cuối, chuỗi ngày): "
        "**SARIMAX** (Auto-ARIMA + nhân tố), **Random Forest / XGBoost** (lag PM2.5 + FA), "
        "**LSTM** một biến, **Hybrid** (SARIMAX + LSTM trên phần dư). "
        "Dữ liệu biểu đồ lấy từ `export_demo_data.py`."
    )

    cmp_df = load_demo_models_comparison()
    metrics_multi = load_demo_models_metrics()
    pred_df = load_demo_predictions()
    params = load_demo_params()

    if cmp_df is None and pred_df is None:
        st.warning("Chưa có dữ liệu dự báo. Chạy: `python export_demo_data.py` rồi mở lại trang này.")
    else:
        # --- Tổng quan chỉ số ---
        st.subheader("📊 Tổng quan so sánh mô hình (test set)")
        if metrics_multi:
            rows = [{"Mô hình": k, **v} for k, v in metrics_multi.items()]
            mdf = pd.DataFrame(rows).sort_values("RMSE", ascending=True)
            st.dataframe(mdf.round({"RMSE": 2, "MAE": 2, "MAPE": 2}), use_container_width=True, hide_index=True)
            fig_bar = px.bar(
                mdf,
                x="Mô hình",
                y="RMSE",
                color="Mô hình",
                title="RMSE theo mô hình (càng thấp càng tốt)",
            )
            fig_bar.update_layout(height=380, xaxis_tickangle=-25, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            best = mdf.iloc[0]["Mô hình"]
            st.success(f"Mô hình RMSE thấp nhất trên file hiện tại: **{best}** ({mdf.iloc[0]['RMSE']:.2f}).")
        elif metrics_multi is None:
            st.info("Chạy `python export_demo_data.py` để tạo `demo_models_metrics.json` (bảng đầy đủ).")

        st.subheader("Tham số SARIMAX (Auto-ARIMA)")
        if params:
            o = params.get("order", [2, 1, 1])
            s = params.get("seasonal_order", [1, 0, 0, 7])
            st.write(
                f"**ARIMA (p,d,q) × (P,D,Q,s)**: ({o[0]},{o[1]},{o[2]}) × ({s[0]},{s[1]},{s[2]},{s[3]}) — mùa vụ tuần **m=7**."
            )
        st.caption("Cùng logic với pipeline `src/sarimax_model.py`.")

        # --- Biểu đồ đa mô hình ---
        st.subheader("Biểu đồ dự báo đa mô hình")
        if cmp_df is not None:
            dates = cmp_df["date"]
            actual = cmp_df["actual"]
            model_options = {
                "SARIMAX (+ FA exog)": ("pred_sarimax", True),
                "Random Forest (lag + FA)": ("pred_rf", False),
                "XGBoost (lag + FA)": ("pred_xgb", False),
                "LSTM (univariate)": ("pred_lstm_uni", False),
                "Hybrid SARIMAX + LSTM (phần dư)": ("pred_hybrid", False),
            }
            chosen = st.multiselect(
                "Hiển thị đường dự báo",
                list(model_options.keys()),
                default=list(model_options.keys()),
            )
            palette = {
                "SARIMAX (+ FA exog)": "#E94F37",
                "Random Forest (lag + FA)": "#44AF69",
                "XGBoost (lag + FA)": "#9B59B6",
                "LSTM (univariate)": "#2CA02C",
                "Hybrid SARIMAX + LSTM (phần dư)": "#F39C12",
            }
            fig_m = go.Figure()
            fig_m.add_trace(
                go.Scatter(
                    x=dates,
                    y=actual,
                    name="Thực tế",
                    line=dict(color="#2E86AB", width=2),
                )
            )
            show_ci = "SARIMAX (+ FA exog)" in chosen
            if show_ci and "ci_low" in cmp_df.columns and "ci_high" in cmp_df.columns:
                fig_m.add_trace(
                    go.Scatter(
                        x=list(dates) + list(dates)[::-1],
                        y=list(cmp_df["ci_high"]) + list(cmp_df["ci_low"])[::-1],
                        fill="toself",
                        fillcolor="rgba(233,79,55,0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="95% CI (SARIMAX)",
                    )
                )
            for label in chosen:
                col, _ = model_options[label]
                if col not in cmp_df.columns:
                    continue
                yv = cmp_df[col]
                fig_m.add_trace(
                    go.Scatter(
                        x=dates,
                        y=yv,
                        name=label,
                        line=dict(color=palette.get(label, "#888"), width=1.8),
                    )
                )
            fig_m.update_layout(
                height=480,
                xaxis_title="Ngày",
                yaxis_title="PM2.5 (μg/m³)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
            )
            st.plotly_chart(fig_m, use_container_width=True)
        elif pred_df is not None:
            st.warning("Chỉ có `demo_predictions.csv` (SARIMAX). Chạy `export_demo_data.py` bản mới để có đủ mô hình.")
            dates = pred_df["date"]
            actual = pred_df["actual"]
            predicted_base = pred_df["predicted"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=actual, name="Thực tế", line=dict(color="#2E86AB", width=2)))
            fig.add_trace(go.Scatter(x=dates, y=predicted_base, name="SARIMAX", line=dict(color="#E94F37", width=2)))
            fig.add_trace(
                go.Scatter(
                    x=list(dates) + list(dates)[::-1],
                    y=list(pred_df["ci_high"]) + list(pred_df["ci_low"])[::-1],
                    fill="toself",
                    fillcolor="rgba(233,79,55,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% CI",
                )
            )
            fig.update_layout(height=450, xaxis_title="Ngày", yaxis_title="PM2.5 (μg/m³)")
            st.plotly_chart(fig, use_container_width=True)

        # --- What-if ---
        st.subheader("What-if: phân tích kịch bản nhân tố")
        st.markdown(
            r"""
            **Cách tính trong app (minh họa, không fit lại SARIMAX):**

            - Trung bình 3 thanh: $\bar{f} = (F_1 + F_2 + F_3) / 3$ (mặc định mỗi thanh = 1 → $\bar{f}=1$).
            - Hệ số nhân lên dự báo SARIMAX gốc $\hat{y}^{\mathrm{sarimax}}$:
            $$k = 0{,}7 + 0{,}3 \times \bar{f}$$
            - Dự báo kịch bản: $\hat{y}^{\mathrm{scenario}} = k \cdot \hat{y}^{\mathrm{sarimax}}$.

            Khi **tất cả thanh = 1** thì $k = 1$: đường kịch bản trùng SARIMAX. Kéo **cùng tăng** (ví dụ $\bar{f}=1{,}2$) thì $k=1{,}06$ — toàn bộ dự báo **cao hơn 6%**; kéo **cùng giảm** ($\bar{f}=0{,}8$) thì $k=0{,}94$ — **thấp hơn 6%**.

            **Lưu ý:** Bảng *Tổng quan so sánh mô hình* phía trên lấy từ file export — **không** đổi theo slider. Chỉ các chỉ số trong khối dưới đây so sánh **thực tế vs SARIMAX gốc** và **thực tế vs đường kịch bản** (cập nhật khi bạn kéo).
            """
        )
        if cmp_df is not None or pred_df is not None:
            if cmp_df is not None:
                dates_w = cmp_df["date"]
                actual_w = cmp_df["actual"]
                pred_w = cmp_df["pred_sarimax"]
            else:
                dates_w = pred_df["date"]
                actual_w = pred_df["actual"]
                pred_w = pred_df["predicted"]
            f1 = st.slider("Factor 1 (ô nhiễm / giao thông)", 0.5, 1.5, 1.0, 0.05)
            f2 = st.slider("Factor 2 (khí hậu)", 0.5, 1.5, 1.0, 0.05)
            f3 = st.slider("Factor 3 (khuếch tán)", 0.5, 1.5, 1.0, 0.05)
            adj = (f1 + f2 + f3) / 3
            k_scale = 0.7 + 0.3 * adj
            pred_adj = pred_w * k_scale
            m_base = compute_error_metrics(actual_w.values, pred_w.values)
            m_scen = compute_error_metrics(actual_w.values, pred_adj.values)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("RMSE — SARIMAX gốc", f"{m_base['RMSE']:.2f}", help="So với PM2.5 thực tế trên tập test.")
                st.metric("RMSE — kịch bản (slider)", f"{m_scen['RMSE']:.2f}", f"{m_scen['RMSE'] - m_base['RMSE']:+.2f} vs gốc")
            with c2:
                st.metric("MAE — SARIMAX gốc", f"{m_base['MAE']:.2f}")
                st.metric("MAE — kịch bản", f"{m_scen['MAE']:.2f}", f"{m_scen['MAE'] - m_base['MAE']:+.2f} vs gốc")
            with c3:
                st.metric("MAPE — SARIMAX gốc", f"{m_base['MAPE']:.1f}%")
                st.metric("MAPE — kịch bản", f"{m_scen['MAPE']:.1f}%", f"{m_scen['MAPE'] - m_base['MAPE']:+.2f} điểm % vs gốc")
            st.caption(f"Hệ số nhân hiện tại **k = {k_scale:.4f}** (trung bình nhân tố **{adj:.3f}**).")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dates_w, y=actual_w, name="Thực tế", line=dict(color="#2E86AB")))
            fig2.add_trace(
                go.Scatter(x=dates_w, y=pred_w, name="SARIMAX gốc (k=1)", line=dict(color="#999", dash="dot"))
            )
            fig2.add_trace(
                go.Scatter(
                    x=dates_w,
                    y=pred_adj,
                    name=f"Kịch bản (F1={f1}, F2={f2}, F3={f3})",
                    line=dict(color="#E94F37"),
                )
            )
            fig2.update_layout(height=350, title="Thực tế vs SARIMAX vs kịch bản (minh họa)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Thiếu dữ liệu cho what-if.")

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
        st.caption("SARIMAX với biến ngoại sinh (Factor Analysis) giúp cải thiện dự báo so với ARIMA/SARIMA thuần.")
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
st.sidebar.caption("PM2.5 | Factor Analysis & SARIMAX | UIT")
