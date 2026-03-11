# BÁO CÁO BỘ DỮ LIỆU VÀ XỬ LÝ NHIỄU

**Dự án:** Time Series Analysis and Forecasting of PM2.5: An SARIMA and Factor Analysis Approach  
**Phiên bản:** 1.0  
**Ngày:** Tháng 3/2025  

---

## 1. GIỚI THIỆU

Báo cáo này mô tả bộ dữ liệu chất lượng không khí được sử dụng trong dự án, nguồn gốc, cấu trúc, và các bước tiền xử lý (preprocessing) đã thực hiện để chuẩn bị dữ liệu cho phân tích nhân tố (Factor Analysis) và mô hình dự báo SARIMA.

---

## 2. NGUỒN DỮ LIỆU

| Thuộc tính | Mô tả |
|------------|-------|
| **Dataset** | PRSA (Beijing Multi-Site Air-Quality Data) |
| **Trạm đo** | Guanyuan (Quan Nguyên, Bắc Kinh) |
| **Khoảng thời gian** | 01/03/2013 – 28/02/2017 |
| **Độ phân giải** | Theo giờ (hourly) |
| **Số quan sát** | ~35.064 dòng (sau khi loại header) |
| **Nguồn gợi ý** | UCI Machine Learning Repository, Kaggle, hoặc trang chính thức của PRSA |

---

## 3. MÔ TẢ CÁC BIẾN SỐ

| Biến | Đơn vị | Mô tả |
|------|--------|-------|
| **PM2.5** | μg/m³ | Nồng độ bụi mịn PM2.5 (biến mục tiêu) |
| **PM10** | μg/m³ | Nồng độ bụi mịn PM10 |
| **SO2** | μg/m³ | Lưu huỳnh đioxit |
| **NO2** | μg/m³ | Nitơ đioxit |
| **CO** | μg/m³ | Cacbon monoxit |
| **O3** | μg/m³ | Ozon |
| **TEMP** | °C | Nhiệt độ |
| **PRES** | hPa | Áp suất khí quyển |
| **DEWP** | °C | Điểm sương |
| **RAIN** | mm | Lượng mưa |
| **wd** | - | Hướng gió (categorical) |
| **WSPM** | m/s | Tốc độ gió |
| **year, month, day, hour** | - | Thành phần thời gian |

---

## 4. XỬ LÝ DỮ LIỆU THIẾU (MISSING VALUES)

### 4.1. Tầm quan trọng

Trong chuỗi thời gian, việc loại bỏ (drop) dòng có giá trị thiếu có thể gây ra:

- Mất tính liên tục của chuỗi thời gian
- Sai lệch thống kê nếu missing không ngẫu nhiên
- Khó áp dụng các mô hình yêu cầu chuỗi đầy đủ (như SARIMA)

Do đó, dự án sử dụng **nội suy (Interpolation)** thay vì loại bỏ dữ liệu thiếu.

### 4.2. Phương pháp đã áp dụng

1. **Nội suy theo thời gian (Temporal interpolation)**  
   - Phương pháp: `interpolate(method="time")`  
   - Ý nghĩa: Nội suy tuyến tính dựa trên index thời gian. Giá trị thiếu được ước lượng từ các điểm lân cận trong chuỗi thời gian.

2. **Xử lý tại biên (Boundary)**  
   - Nếu sau bước 1 vẫn còn giá trị thiếu (ví dụ đầu/cuối chuỗi):  
     - `ffill()`: Forward fill  
     - `bfill()`: Backward fill  

### 4.3. Phạm vi áp dụng

- Chỉ áp dụng cho **các biến số (numeric)**.
- Các biến phân loại (ví dụ: `wd`, `station`) không được nội suy; nếu cần dùng trong mô hình, sẽ được mã hóa riêng hoặc loại trừ.

---

## 5. QUY TRÌNH TIỀN XỬ LÝ (PREPROCESSING PIPELINE)

Toàn bộ quy trình tiền xử lý được thực hiện trong **`notebooks/01_EDA.ipynb`** (gộp EDA và data prep).

1. **Đọc dữ liệu**  
   - Đọc file CSV từ `data/raw/` (file đầu tiên có đuôi .csv).

2. **Tạo index thời gian**  
   - Gộp các cột `year`, `month`, `day`, `hour` thành datetime và đặt làm index; sắp xếp theo thời gian.  
   - Nếu không có các cột trên: dùng cột `datetime`, `date`, `Date` hoặc `timestamp` (nếu có).

3. **Xử lý missing values**  
   - Nội suy theo thời gian (`interpolate(method="time")`) cho tất cả biến số.  
   - Forward/backward fill cho các vị trí còn thiếu.

4. **Lưu dữ liệu đã làm sạch**  
   - Xuất ra `data/interim/cleaned_data.csv` (sau khi bỏ cột phụ chỉ dùng cho EDA như month, hour).

---

## 6. KẾT QUẢ EDA VÀ BIỂU ĐỒ

Tất cả biểu đồ EDA được tạo trong **`notebooks/01_EDA.ipynb`** và lưu tại `reports/figures/01_eda/`.

- **Missing value summary:** `missing_summary.png` – bảng và biểu đồ % missing theo cột.
- **PM2.5 – boxplot & distribution:** `pm25_boxplot_distribution.png`.
- **PM2.5 theo tháng:** `pm25_by_month.png`, `pm25_box_by_month.png`.
- **PM2.5 theo giờ:** `pm25_by_hour.png` – trung bình ± std theo giờ trong ngày.
- **Rolling mean / std:** `pm25_rolling_mean_std.png` – cửa sổ 7 ngày.
- **ACF / PACF (PM2.5 hourly):** `acf_pacf_pm25.png`.
- **Histogram các biến chính:** `histograms_main_vars.png`.
- **Scatter PM2.5 vs biến:** `scatter_pm25_vs_vars.png`.
- **KMO & Bartlett:** in trong notebook – kiểm tra phù hợp dữ liệu cho Factor Analysis.
- **Decomposition & seasonal:** `decomposition_pm25_daily.png`, `seasonal_plot_pm25_by_month.png`.
- **Chuỗi PM2.5 và ma trận tương quan:** `pm25_time_series.png`, `correlation_heatmap.png`.

---

## 7. KẾT QUẢ PHÂN TÍCH NHÂN TỐ (FACTOR ANALYSIS)

| File | Mô tả |
|------|-------|
| `reports/tables/factor_loadings.csv` | Ma trận Factor Loadings (biến ↔ nhân tố) |
| `reports/tables/factor_interpretation.txt` | Kết luận các nhóm nguyên nhân chính ảnh hưởng PM2.5 |
| `reports/figures/02_fa/factor_loadings_heatmap.png` | Heatmap trực quan hóa loadings |
| `reports/figures/02_fa/scree_plot.png` | Scree plot (eigenvalues, Kaiser criterion) |

---

## 8. TÍNH MÙA VỤ CỦA PM2.5 VÀ CÁCH ĐỌC ĐỒ THỊ ACF/PACF

### 8.1. Tính mùa vụ (Seasonality)

**Tính mùa vụ** là sự lặp lại theo chu kỳ của chuỗi thời gian. Đối với PM2.5:

- **Chu kỳ theo ngày (daily):** Dữ liệu gốc theo giờ, sau khi gộp theo ngày ta có chuỗi daily. Chu kỳ mùa vụ thường gặp là **tuần (m=7)**: hoạt động con người (giao thông, sản xuất) thay đổi theo ngày trong tuần (cuối tuần khác ngày thường).
- **Chu kỳ theo mùa (annual):** PM2.5 thường cao hơn vào mùa đông (sưởi ấm, nghịch nhiệt) và thấp hơn vào mùa hè (mưa, gió).

**Phân rã chuỗi thời gian (Time Series Decomposition)** tách PM2.5 thành:
- **Trend (Xu hướng):** biến động dài hạn
- **Seasonal (Mùa vụ):** biến động lặp theo chu kỳ
- **Residual (Phần dư):** nhiễu ngẫu nhiên

Trong dự án, phân rã dùng **period = 7** (chu kỳ tuần) cho dữ liệu daily. Biểu đồ: `reports/figures/03_arima/decomposition.png`.

### 8.2. Cách đọc đồ thị ACF và PACF

**ACF (Autocorrelation Function)** – Hàm tự tương quan:
- Trục hoành: **lag** (độ trễ, số bước thời gian)
- Trục tung: hệ số tương quan giữa chuỗi và phiên bản trễ của nó
- **Cách đọc:** Thanh vượt qua vùng tin cậy (dải xanh) cho thấy có tương quan đáng kể tại lag đó. ACF suy giảm chậm → có xu hướng (trend); ACF có đỉnh tại lag 7, 14, … → có mùa vụ tuần.

**PACF (Partial Autocorrelation Function)** – Hàm tự tương quan riêng phần:
- Đo tương quan giữa \(y_t\) và \(y_{t-k}\) sau khi loại bỏ ảnh hưởng của các lag trung gian
- **Cách đọc:** Thanh vượt vùng tin cậy → lag đó đóng góp độc lập. PACF cắt đứt (nhiều lag ~ 0) sau vài lag → gợi ý bậc **p** cho phần AR.

**Ứng dụng cho SARIMA (p,d,q)(P,D,Q,m):**
- **ACF:** Gợi ý bậc **q** (MA) và **Q** (seasonal MA). Đỉnh tại lag s, 2s, … → Q > 0.
- **PACF:** Gợi ý bậc **p** (AR) và **P** (seasonal AR). Đỉnh tại lag s, 2s, … → P > 0.
- **m:** chu kỳ mùa vụ (m=7 cho daily, m=12 cho monthly).

Biểu đồ ACF/PACF: `reports/figures/03_arima/acf_pacf.png`. Dự án dùng **auto_arima** để tìm bộ tham số tối ưu dựa trên AIC/BIC, kết quả: (2,1,1)×(1,0,0,7).

---

## 9. FILE CODE LIÊN QUAN

| Bước | File | Mô tả |
|------|------|-------|
| Tiền xử lý + EDA | `notebooks/01_EDA.ipynb` | Đọc raw, datetime index, EDA đầy đủ, xử lý missing, lưu cleaned_data + 01_eda figures |
| Pipeline chính | `main.py` | Kiểm tra cleaned_data có sẵn, gọi FA → SARIMA → Evaluation |
| Phân tích nhân tố | `src/factor_analysis.py` | Đọc cleaned_data, FA (principal, varimax), scree plot, lưu fa_data.csv + 02_fa + tables |
| Mô hình SARIMA | `src/sarima_model.py` | Đọc fa_data, gộp daily, decomposition, ADF, ACF/PACF, auto_arima, lưu sarima_model.joblib + 03_arima |
| Đánh giá | `src/evaluation.py` | Đọc model + fa_data, dự báo test, RMSE/MAE/MAPE, residual diagnostics, lưu 04_eval + demo_* |
| Dữ liệu demo (dashboard) | `export_demo_data.py` | Xuất demo_predictions.csv, demo_residuals.csv, demo_params.json cho app |
| Dashboard | `app.py` | Streamlit: 4 trang (Tổng quan, Nhân tố, Dự báo, Đánh giá) |

---

## 10. TỔNG HỢP FA + SARIMA VÀ ĐỀ XUẤT GIẢI PHÁP

### Kết nối hai phần

**Phân tích nhân tố (FA)** cho thấy ba nhóm yếu tố chính ảnh hưởng đến PM2.5:

1. **Factor 1 – Ô nhiễm phát thải:** PM10, CO, SO2, NO2 (giao thông, công nghiệp, đốt nhiên liệu).
2. **Factor 2 – Khí tượng:** TEMP, PRES, DEWP, O3 (nhiệt độ, áp suất, điều kiện khí quyển).
3. **Factor 3 – Khuếch tán:** WSPM, O3, NO2 (tốc độ gió, phản ứng quang hóa).

**Mô hình SARIMA** (2,1,1)×(1,0,0,7) dùng ba nhân tố trên làm biến ngoại sinh (exog) để dự báo PM2.5. Phân rã chuỗi cho thấy tính mùa vụ tuần (m=7), phù hợp nhịp hoạt động giao thông, công nghiệp theo ngày trong tuần.

**Đánh giá:** RMSE ≈ 37.8, MAE ≈ 30.1, MAPE ≈ 69.7% trên tập test. Residual diagnostics (Ljung-Box, Q-Q) cho thấy mô hình bắt được phần lớn cấu trúc chuỗi thời gian.

### Đề xuất giải pháp thực tiễn

- **Khi FA chỉ ra Factor 1 (PM10, CO, SO2, NO2) mạnh và SARIMA dự báo PM2.5 tăng:** Ưu tiên giảm phát thải giao thông và công nghiệp (giảm xe cá nhân, kiểm soát khí thải, hạn chế đốt than).
- **Khi Factor 2 (khí tượng) bất lợi:** Cảnh báo sớm khi nhiệt độ thấp, áp suất cao, ít gió (dễ tích tụ ô nhiễm). Khuyến cáo hạn chế hoạt động ngoài trời.
- **Khi Factor 3 (tốc độ gió) thấp:** Giảm khuếch tán, PM2.5 dễ tích tụ. Kết hợp với dự báo SARIMA để phát cảnh báo chất lượng không khí theo ngày.
- **Theo mùa vụ:** Dự báo PM2.5 tăng vào mùa đông → tăng kiểm tra, giám sát sưởi ấm và đốt nhiên liệu rắn.

---

## 11. SƯỜN BÁO CÁO WORD (CUỐI KỲ)

1. **Introduction (Mở đầu)**
   - Bối cảnh ô nhiễm không khí và PM2.5
   - Mục tiêu nghiên cứu: phân tích nhân tố + dự báo chuỗi thời gian
   - Cấu trúc báo cáo

2. **Literature Review (Tổng quan nghiên cứu)**
   - Các nghiên cứu về PM2.5 và chất lượng không khí
   - Factor Analysis / PCA trong nghiên cứu môi trường
   - SARIMA và mô hình dự báo chuỗi thời gian

3. **Methodology (Phương pháp)**
   - Nguồn dữ liệu và mô tả biến
   - Tiền xử lý trong EDA notebook (missing values, interpolation theo thời gian)
   - Factor Analysis: phương pháp, số nhân tố, phép quay
   - SARIMA: phân rã chuỗi, ADF, ACF/PACF, auto_arima
   - Đánh giá: RMSE, MAE, MAPE, residual diagnostics

4. **Results (Kết quả)**
   - EDA và biểu đồ tương quan
   - Kết quả FA: factor loadings, diễn giải nhân tố
   - Kết quả SARIMA: tham số tối ưu, biểu đồ decomposition, ACF/PACF
   - Biểu đồ thực tế vs dự báo
   - Bảng chỉ số đánh giá
   - Tổng hợp FA + SARIMA → đề xuất

5. **Conclusion (Kết luận)**
   - Tóm tắt kết quả chính
   - Hạn chế nghiên cứu
   - Hướng phát triển

6. **References (Tài liệu tham khảo)**

---

## 12. DÀN Ý SLIDE THUYẾT TRÌNH

| Slide | Nội dung | Ghi chú |
|-------|----------|---------|
| 1 | Title slide: PM2.5 – Factor Analysis & SARIMA | |
| 2 | Vấn đề: Ô nhiễm không khí, PM2.5 và sức khỏe | Đặt bối cảnh |
| 3 | Mục tiêu: Phân tích nguyên nhân + Dự báo PM2.5 | |
| 4 | Nguồn dữ liệu: PRSA Beijing, trạm Guanyuan | |
| 5 | Tiền xử lý: Interpolation thay vì drop | |
| 6 | EDA: Biểu đồ PM2.5, ma trận tương quan | Hình 01_eda |
| 7 | **Factor Analysis:** Scree plot, số nhân tố | Hình 02_fa |
| 8 | **Factor Analysis:** Loadings, 3 nhóm nguyên nhân chính | Bảng + heatmap |
| 9 | **SARIMA:** Phân rã chuỗi – Trend, Seasonal, Residual | Hình decomposition |
| 10 | **SARIMA:** ACF/PACF, cách đọc, chọn tham số | Hình acf_pacf |
| 11 | **SARIMA:** Mô hình (2,1,1)×(1,0,0,7), exog = Factors | |
| 12 | **Đánh giá:** RMSE, MAE, MAPE; Actual vs Predicted | Bảng + hình 04_eval |
| 13 | **Đánh giá:** Residual diagnostics (Ljung-Box, Q-Q) | Hình residual_diagnostics |
| 14 | **Tổng hợp:** FA → SARIMA → Đề xuất giải pháp thực tiễn | 3–4 bullet |
| 15 | Kết luận & Q&A | |

**Flow logic:** Vấn đề ô nhiễm → Nguyên nhân (FA) → Dự báo (SARIMA) → Đánh giá độ tin cậy → Đề xuất giải pháp.

---

## 13. KẾT LUẬN

Bộ dữ liệu PRSA tại trạm Guanyuan đã được tiền xử lý theo quy trình chuẩn, trong đó việc xử lý dữ liệu thiếu bằng **Interpolation** là bước quan trọng để duy trì chuỗi thời gian liên tục phục vụ mô hình SARIMA và phân tích nhân tố. Báo cáo này cung cấp sườn nội dung và dàn ý slide để triển khai báo cáo Word và bài thuyết trình cuối kỳ.

*Báo cáo này có thể xuất sang Word/PDF bằng công cụ như Pandoc: `pandoc DATA_REPORT.md -o DATA_REPORT.docx`*
