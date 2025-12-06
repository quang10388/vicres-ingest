\# VICRES-INGEST



Công cụ thu nhận \& xử lý dữ liệu đầu vào (DEM, mưa, nhiệt độ, gió) cho mô hình thủy văn VIC-Res.



\## 1. Cài đặt môi trường



```bash

conda env create -f envs/environment.yml

conda activate vicres-ingest

pip install -e .

\#Nếu đã có sẵn môi trường, chỉ cần pip install -e .

2\. Chuẩn bị dữ liệu \& cấu hình



Đặt DEM, shapefile lưu vực và các dữ liệu liên quan theo cấu trúc thư mục chuẩn

(xem thư mục ref/ và data/ mẫu).



Copy file params.template.yaml thành params.yaml:



copy params.template.yaml params.yaml





Mở params.yaml và chỉnh:



project.root: "."



Đường dẫn DEM, shapefile lưu vực



Thời kỳ xử lý (t0/t1 cho mưa, years/months cho nhiệt độ, gió)



Nguồn dữ liệu (CHIRPS/IMERG, ERA5/ERA5-Land, backend cds/local,...).



3\. Chạy các bước chính (CLI)

vicres-ingest build-grid   -p params.yaml    # DEM → lưới VIC + template NC

vicres-ingest precip       -p params.yaml    # CHIRPS/IMERG → mưa VIC-Res

vicres-ingest temperature  -p params.yaml    # ERA5(-Land) → Tmin/Tmax VIC-Res

vicres-ingest wind         -p params.yaml    # ERA5(-Land) → gió VIC-Res





Các hàm tương ứng nằm trong module vicres\_tool.



Kết quả điển hình:



Lưới \& template VIC-Res: outputs/grids/



Forcing NetCDF: outputs/forcings/netcdf/



Forcing ASCII (VIC-Res): outputs/forcings/ascii/



Hình QC: outputs/figs/ (mưa, nhiệt độ, gió)

