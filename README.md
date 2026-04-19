# PM2.5 Time Series: Factor Analysis & SARIMA

**Môn học:** Phân tích dữ liệu lớn — IT2036-CH201  
**Đề tài:** Time Series Analysis and Forecasting of PM2.5 — SARIMA và Phân tích nhân tố  

Dự án phân tích chuỗi thời gian nồng độ **PM2.5** (bụi mịn), kết hợp **phân tích nhân tố (FA)** để rút gọn biến môi trường/khí tượng và mô hình **SARIMA** có **biến ngoại sinh** (điểm số nhân tố). Pipeline gồm tiền xử lý, ước lượng mô hình, đánh giá và **dashboard Streamlit** tương tác.

---

## Nội dung

- [Tính năng chính](#tính-năng-chính)
- [Công nghệ](#công-nghệ)
- [Cài đặt](#cài-đặt)
- [Dữ liệu đầu vào](#dữ-liệu-đầu-vào)
- [Luồng chạy chuẩn](#luồng-chạy-chuẩn)
- [Notebook bổ sung](#notebook-bổ-sung)
- [Dashboard](#dashboard)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Phương pháp (tóm tắt)](#phương-pháp-tóm-tắt)
- [Kết quả tham chiếu](#kết-quả-tham-chiếu)
- [Tài liệu tham khảo trong repo](#tài-liệu-tham-khảo-trong-repo)

---

## Tính năng chính

| Hạng mục | Mô tả |
|----------|--------|
| **Phân tích nhân tố** | Chuẩn hóa 10 biến (PM10, SO₂, NO₂, CO, O₃, TEMP, PRES, DEWP, RAIN, WSPM), FA (principal + varimax), chọn số nhân tố theo **Kaiser** (mặc định 3 nhân tố) |
| **SARIMA / SARIMAX** | Gộp **theo ngày**, `auto_arima` (pmdarima), mùa vụ **tuần** `m=7`, exog = Factor1–3 |
| **Đánh giá** | RMSE, MAE, MAPE; biểu đồ actual vs predicted; chẩn đoán phần dư (histogram, Q-Q, Ljung-Box) |
| **Ứng dụng web** | Streamlit: khám phá dữ liệu, nhân tố, dự báo và what-if trên nhân tố |

---

## Công nghệ

- **Python 3.8+** (khuyến nghị 3.10+)
- **pandas**, **numpy** — xử lý dữ liệu  
- **factor_analyzer**, **scikit-learn** — phân tích nhân tố  
- **pmdarima**, **statsmodels** — SARIMA và kiểm định chuỗi  
- **matplotlib**, **seaborn** — hình ảnh báo cáo  
- **streamlit**, **plotly** — dashboard  
- **tensorflow** (tùy chọn) — LSTM trong `src/lstm_model.py` và notebook mở rộng  
- **xgboost** (tùy chọn) — baseline trong `notebooks/05_multi_scene.ipynb`

Chi tiết phiên bản: `requirements.txt`.

---

## Cài đặt

```bash
git clone <repository-url>
cd pm25_sarima_project

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dữ liệu đầu vào

- **Nguồn:** PRSA (Beijing Multi-Site Air-Quality Data), trạm **Guanyuan**.  
- **Thời gian:** 01/03/2013 — 28/02/2017, độ phân giải **theo giờ**.  
- **Đặt file CSV gốc** vào `data/raw/` (ví dụ `PRSA_Data_Guanyuan_20130301-20170228.csv`).  

Cột tối thiểu cần có trong pipeline hiện tại: `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`, `wd`, `WSPM`, và các trường thời gian (`year`, `month`, `day`, `hour` hoặc tương đương để tạo chỉ mục thời gian — xem `notebooks/01_EDA.ipynb`).

---

## Luồng chạy chuẩn

Thứ tự bắt buộc: **EDA notebook** → **`main.py`** → (tùy chọn) **`export_demo_data.py`** → **`streamlit run app.py`**.

### Bước 1 — EDA và tiền xử lý (bắt buộc)

```text
notebooks/01_EDA.ipynb
```

Sinh ra: `data/interim/cleaned_data.csv`, hình trong `reports/figures/01_eda/`.  
Không có `cleaned_data.csv` thì `main.py` sẽ dừng và báo lỗi.

### Bước 2 — Pipeline tự động

```bash
python main.py
```

| Bước | Mô-đun | Đầu ra chính |
|------|--------|----------------|
| 2 | `src/factor_analysis.py` | `data/processed/fa_data.csv`, `reports/figures/02_fa/`, bảng loadings |
| 3 | `src/sarima_model.py` | `data/processed/sarima_model.joblib`, `reports/figures/03_arima/` |
| 4 | `src/evaluation.py` | `reports/figures/04_eval/`, `reports/tables/evaluation_metrics.csv`, `data/processed/demo_*.csv` |

Chạy từng bước riêng (sau khi đã có `cleaned_data.csv`):

```bash
python src/factor_analysis.py
python src/sarima_model.py
python src/evaluation.py
```

---

## Notebook bổ sung

| Notebook | Mục đích |
|----------|-----------|
| `notebooks/02_factor_analysis.ipynb` | Đi sâu bước FA (song song với `factor_analysis.py`) |
| `notebooks/03_sarima_model.ipynb` | Đi sâu SARIMA (song song với `sarima_model.py`) |
| `notebooks/04_evaluation.ipynb` | Đi sâu đánh giá (song song với `evaluation.py`) |
| `notebooks/05_multi_scene.ipynb` | **Mở rộng:** so sánh SARIMAX, Random Forest / XGBoost (lag + nhân tố), LSTM, hybrid SARIMAX + LSTM trên phần dư; đọc `data/processed/fa_data.csv` trong repo |

Bản đã chạy sẵn (tham khảo output): `notebooks/execute/*_executed.ipynb`.

---

## Dashboard

1. Tạo file demo cho app (tránh lỗi khi load trực tiếp model pickle/joblib trên một số môi trường):

   ```bash
   python export_demo_data.py
   ```

2. Chạy giao diện:

   ```bash
   streamlit run app.py
   ```

Mở trình duyệt tại `http://localhost:8501`.

**Các trang chính:** Tổng quan & khám phá → Phân tích nhân tố → Dự báo & kịch bản → Đánh giá & chẩn đoán phần dư.

---

## Cấu trúc thư mục

```text
pm25_sarima_project/
├── main.py                 # Pipeline: FA → SARIMA → Evaluation
├── app.py                  # Dashboard Streamlit
├── export_demo_data.py     # Xuất CSV demo cho dashboard
├── requirements.txt
├── README.md
│
├── notebooks/
│   ├── 01_EDA.ipynb        # Bước 1: EDA + cleaned_data (bắt buộc trước main.py)
│   ├── 02_factor_analysis.ipynb
│   ├── 03_sarima_model.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_multi_scene.ipynb
│   └── execute/            # Bản notebook đã execute (tham khảo)
│
├── src/
│   ├── factor_analysis.py
│   ├── sarima_model.py
│   ├── evaluation.py
│   └── lstm_model.py       # Tiện ích LSTM (pipeline phụ / thử nghiệm)
│
├── data/
│   ├── raw/                # CSV gốc (PRSA)
│   ├── interim/            # cleaned_data.csv
│   └── processed/          # fa_data.csv, model, demo_*.csv
│
└── reports/
    ├── DATA_REPORT.md      # Báo cáo chi tiết dữ liệu & phương pháp
    ├── tables/             # metrics, loadings, ADF, v.v.
    └── figures/
        ├── 01_eda/
        ├── 02_fa/
        ├── 03_arima/
        └── 04_eval/
```

---

## Phương pháp (tóm tắt)

1. **EDA:** làm sạch, nội suy missing theo thời gian, thống kê mô tả và hình ảnh hóa.  
2. **FA:** giảm chiều biến môi trường → Factor1–3 làm biến ngoại sinh cho chuỗi ngày.  
3. **SARIMAX:** `auto_arima` với mùa vụ 7 ngày; thứ tự mô hình điển hình do thuật toán chọn (ví dụ đã gặp **(2,1,1)×(1,0,0,7)** với exog Factor1–3).  
4. **Đánh giá:** tách train/test theo thời gian (tỷ lệ test mặc định trong code), chỉ số và kiểm tra phần dư.

Chi tiết định nghĩa biến, hình và bảng: **`reports/DATA_REPORT.md`**.

---

## Kết quả tham chiếu

Giá trị phụ thuộc seed / phiên bản thư viện; tham khảo file sinh bởi pipeline:

- **`reports/tables/evaluation_metrics.csv`** — RMSE, MAE, MAPE trên tập test.  
- Ví dụ đã ghi nhận: RMSE ≈ **37.8**, MAE ≈ **30.1**, MAPE ≈ **69.7%** (daily, test ~20%).  
- Phần dư: kiểm tra Ljung-Box với p-value > 0.05 tại các lag xét (mục tiêu: không còn tự tương quan rõ).

---

## Tài liệu tham khảo trong repo

- **`reports/DATA_REPORT.md`** — Báo cáo dữ liệu, phương pháp và gợi ý trình bày báo cáo/slide.

---

**UIT** | Phân tích dữ liệu lớn — IT2036-CH201 | Đồ án cuối kỳ
