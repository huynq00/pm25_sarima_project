# PM2.5 Time Series Analysis: Factor Analysis & SARIMA

**Môn học:** Phân tích dữ liệu lớn - IT2036-CH201  
**Dự án cuối kỳ:** Time Series Analysis and Forecasting of PM2.5: An SARIMA and Factor Analysis Approach  

Dự án phân tích chuỗi thời gian và dự báo nồng độ PM2.5 (bụi mịn) kết hợp **Phân tích Nhân tố (Factor Analysis)** và mô hình **SARIMA** với biến ngoại sinh (exogenous variables). Bao gồm pipeline xử lý dữ liệu, mô hình dự báo, đánh giá và dashboard tương tác bằng Streamlit.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Cài đặt](#cài-đặt)
- [Chạy Pipeline](#chạy-pipeline)
- [Dashboard Web](#dashboard-web)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Phương pháp nghiên cứu](#phương-pháp-nghiên-cứu)
- [Nguồn dữ liệu](#nguồn-dữ-liệu)
- [Kết quả chính](#kết-quả-chính)

---

## Tổng quan

### Mục tiêu

- **Phân tích nhân tố:** Rút gọn 10 biến môi trường/khí tượng (PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM) thành 3 nhân tố tiềm ẩn giải thích nguyên nhân ảnh hưởng đến PM2.5.
- **Dự báo chuỗi thời gian:** Xây dựng mô hình SARIMA với biến ngoại sinh (3 nhân tố) để dự báo PM2.5.
- **Đánh giá:** Đo lường RMSE, MAE, MAPE và kiểm định phần dư (Ljung-Box, Q-Q plot).

### Công nghệ sử dụng

- **Python 3.8+**
- **pandas, numpy** – Xử lý dữ liệu
- **factor_analyzer, scikit-learn** – Phân tích nhân tố
- **pmdarima, statsmodels** – Mô hình SARIMA và phân tích chuỗi thời gian
- **streamlit, plotly** – Dashboard tương tác

---

## Cài đặt

### 1. Tạo môi trường ảo và cài đặt phụ thuộc

```bash
cd pm25_sarima_project
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Hoặc: .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Dữ liệu đầu vào

Đặt file CSV dữ liệu gốc tại `data/raw/`. Dự án sử dụng **PRSA (Beijing Multi-Site Air-Quality Data)** – trạm **Guanyuan**, Bắc Kinh:

- **Khoảng thời gian:** 01/03/2013 – 28/02/2017  
- **Độ phân giải:** theo giờ (hourly)  
- **Cột cần có:** `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`, `wd`, `WSPM`, `year`, `month`, `day`, `hour`

File mẫu: `data/raw/PRSA_Data_Guanyuan_20130301-20170228.csv`  
Nguồn tham khảo: UCI Machine Learning Repository, Kaggle, PRSA.

---

## Chạy Pipeline

Chạy toàn bộ pipeline từ tiền xử lý đến đánh giá:

```bash
python main.py
```

Pipeline thực hiện 4 bước:

| Bước | Mô-đun | Mô tả |
|------|--------|-------|
| 1 | `src/data_prep.py` | Đọc dữ liệu, tạo datetime index, xử lý missing (interpolation), lưu EDA |
| 2 | `src/factor_analysis.py` | FA (principal, varimax), Scree plot, trích xuất 3 nhân tố |
| 3 | `src/sarima_model.py` | Gộp theo ngày, phân rã chuỗi, ADF, ACF/PACF, auto_arima |
| 4 | `src/evaluation.py` | Dự báo test, tính RMSE/MAE/MAPE, residual diagnostics |

### Chạy từng bước riêng lẻ

```bash
python src/data_prep.py
python src/factor_analysis.py
python src/sarima_model.py
python src/evaluation.py
```

---

## Dashboard Web

Ứng dụng Streamlit cung cấp giao diện tương tác với kết quả phân tích và dự báo.

### Chuẩn bị dữ liệu demo

Trước khi chạy dashboard, cần tạo dữ liệu demo (tránh lỗi load model trực tiếp):

```bash
python export_demo_data.py
```

### Chạy dashboard

```bash
streamlit run app.py
```

Trình duyệt mở tại `http://localhost:8501`.

### Cấu trúc 4 trang dashboard

1. **Tổng quan & Khám phá** – Bản đồ trạm, Gauge PM2.5, lịch sử PM2.5, ma trận tương quan  
2. **Khai phá Nhân tố** – Scree plot, bảng Factor Loadings, heatmap  
3. **Dự báo & Phân tích Kịch bản** – Tham số SARIMA, biểu đồ dự báo (95% CI), What-if (điều chỉnh nhân tố)  
4. **Đánh giá & Chẩn đoán** – RMSE/MAE/MAPE, Residual diagnostics (histogram, Q-Q, Ljung-Box)

---

## Cấu trúc dự án

```
pm25_sarima_project/
├── main.py                 # Điểm vào pipeline chính
├── app.py                  # Streamlit dashboard
├── export_demo_data.py     # Xuất dữ liệu demo cho dashboard
├── requirements.txt        # Phụ thuộc Python
├── README.md
├── DEMO_WEB.md             # Hướng dẫn chạy dashboard
│
├── src/
│   ├── data_prep.py        # Bước 1: Tiền xử lý dữ liệu
│   ├── factor_analysis.py  # Bước 2: Phân tích nhân tố
│   ├── sarima_model.py     # Bước 3: Mô hình SARIMA
│   └── evaluation.py       # Bước 4: Đánh giá mô hình
│
├── data/
│   ├── raw/                # Dữ liệu gốc (CSV)
│   ├── interim/            # cleaned_data.csv
│   └── processed/          # fa_data.csv, sarima_model.joblib, demo_*.csv/json
│
└── reports/
    ├── DATA_REPORT.md      # Báo cáo chi tiết dữ liệu và phương pháp
    ├── tables/             # factor_loadings.csv, evaluation_metrics.csv, adf_results.*
    └── figures/
        ├── 01_eda/         # pm25_time_series.png, correlation_heatmap.png
        ├── 02_fa/          # scree_plot.png, factor_loadings_heatmap.png
        ├── 03_arima/       # decomposition.png, acf_pacf.png
        └── 04_eval/        # actual_vs_predicted.png, residual_diagnostics.png
```

---

## Phương pháp nghiên cứu

### 1. Tiền xử lý (`data_prep.py`)

- Tạo datetime index từ `year`, `month`, `day`, `hour`
- Xử lý missing: **interpolation theo thời gian** (thay vì drop) để giữ tính liên tục
- Forward/backward fill cho các vị trí biên
- Xuất biểu đồ EDA: chuỗi PM2.5 theo thời gian, ma trận tương quan

### 2. Phân tích nhân tố (`factor_analysis.py`)

- Chuẩn hóa (z-score) các biến PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM
- Phân tích nhân tố: method `principal`, rotation `varimax`
- Chọn số nhân tố bằng **Kaiser criterion** (eigenvalue > 1), mặc định 3 nhân tố
- Xuất: `factor_loadings.csv`, Scree plot, heatmap loadings, file diễn giải

**Ý nghĩa 3 nhân tố:**

| Nhân tố | Nhóm | Biến chính |
|---------|------|------------|
| Factor 1 | Ô nhiễm phát thải (giao thông, công nghiệp) | PM10, CO, SO2, NO2 |
| Factor 2 | Khí tượng | TEMP, PRES, DEWP, O3 |
| Factor 3 | Khuếch tán | WSPM, O3, NO2 |

### 3. Mô hình SARIMA (`sarima_model.py`)

- Gộp dữ liệu theo ngày (mean)
- Phân rã chuỗi: Trend, Seasonal (period=7), Residual
- Kiểm định ADF (stationarity)
- Đồ thị ACF/PACF để hỗ trợ chọn tham số
- **auto_arima** (pmdarima): tìm bộ tham số tối ưu theo AIC/BIC
- Mô hình: **SARIMA (2,1,1)×(1,0,0,7)** với exog = Factor1, Factor2, Factor3

### 4. Đánh giá (`evaluation.py`)

- Chỉ số: **RMSE**, **MAE**, **MAPE**
- Biểu đồ Actual vs Predicted
- Residual diagnostics: histogram, Q-Q plot, kiểm định Ljung-Box (phần dư không còn tự tương quan)

---

## Nguồn dữ liệu

| Thuộc tính | Mô tả |
|------------|-------|
| Dataset | PRSA (Beijing Multi-Site Air-Quality Data) |
| Trạm đo | Guanyuan (Quan Nguyên, Bắc Kinh) |
| Khoảng thời gian | 01/03/2013 – 28/02/2017 |
| Độ phân giải | Theo giờ |
| Biến chính | PM2.5 (mục tiêu), PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM |

---

## Kết quả chính

- **Mô hình:** SARIMA (2,1,1)×(1,0,0,7) với 3 nhân tố làm biến ngoại sinh
- **Đánh giá (test set):** RMSE ≈ 37.8, MAE ≈ 30.1, MAPE ≈ 69.7%
- **Chu kỳ mùa vụ:** 7 ngày (tuần) – phù hợp nhịp hoạt động giao thông, công nghiệp
- **Residual diagnostics:** Phần dư gần chuẩn, không còn tự tương quan (Ljung-Box p-value > 0.05)

Chi tiết thêm: xem `reports/DATA_REPORT.md`.

---

## Tài liệu tham khảo

- `reports/DATA_REPORT.md` – Báo cáo dữ liệu, phương pháp và dàn ý báo cáo/slide
- `DEMO_WEB.md` – Hướng dẫn chạy dashboard và quay video demo

---

**UIT** | Phân tích dữ liệu lớn - IT2036-CH201 | Dự án cuối kỳ
