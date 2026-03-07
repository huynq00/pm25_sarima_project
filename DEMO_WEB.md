# Dashboard Demo - Chạy Web

## Chuẩn bị dữ liệu (chạy 1 lần)

Nếu gặp lỗi joblib/pandas khi load model, chạy script export để tạo file CSV:

```bash
cd pm25_sarima_project
source .venv/bin/activate
python export_demo_data.py
```

## Chạy ứng dụng Streamlit

```bash
cd pm25_sarima_project
source .venv/bin/activate
pip install streamlit plotly
streamlit run app.py
```

Trình duyệt sẽ mở tại `http://localhost:8501`.

## Cấu trúc 4 trang

1. **Tổng quan & Khám phá** – Bản đồ trạm, Gauge PM2.5, biểu đồ lịch sử, ma trận tương quan  
2. **Khai phá Nhân tố** – Scree plot, bảng Factor Loadings, biểu đồ cột  
3. **Dự báo & Phân tích Kịch bản** – Tham số SARIMA, biểu đồ dự báo (95% CI), What-if sliders  
4. **Đánh giá & Chẩn đoán** – RMSE/MAE/MAPE, Residual diagnostics  

## Kịch bản quay Video Demo

- **Thành viên 1:** Giới thiệu dữ liệu ở Trang 1 & 2  
- **Thành viên 2:** Trang 3 – Bấm Dự báo, điều chỉnh thanh trượt What-if  
- **Thành viên 3:** Trang 4 – Trình bày các chỉ số và residual diagnostics  
