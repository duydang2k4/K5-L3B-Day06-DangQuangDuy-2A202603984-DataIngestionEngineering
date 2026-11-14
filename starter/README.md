# Starter · PostgreSQL + pgvector

Bộ khởi động **tùy chọn** cho lab cá nhân. Docker Compose tạo PostgreSQL + pgvector local; `src/` chứa pipeline Python đọc PDF/DOCX, tạo embedding local, lưu/tìm vector và chạy benchmark. Vẫn có thể chọn Qdrant, Chroma hoặc vector store tương đương theo [GUIDE.md](../GUIDE.md).

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # psycopg, pypdf và python-dotenv
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/check_setup.py # kiểm tra kết nối và extension (đã chạy được)
├── src/
│   ├── models.py          # hợp đồng DocumentPage / Chunk
│   ├── ingestion.py       # đọc PDF/DOCX, chuẩn hóa và giữ trang nguồn
│   ├── chunking.py        # fixed-size và paragraph chunking
│   ├── embeddings.py      # local Sentence Transformers adapter
│   ├── store.py           # pgvector upsert, cosine search và metadata filter
│   ├── evaluation.py      # Hit@1, Evidence Hit@3 và trace metrics
│   └── run.py             # orchestration, manual-review samples và results
├── data/                  # bạn tự chuẩn bị corpus được phép sử dụng
└── results/               # lưu kết quả của bạn
```

## 1. Cài và kiểm tra database

Cần Docker Desktop hoặc Docker Engine + Compose, Python **3.11–3.14**, và mạng để tải image/dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Kỳ vọng: hiện tên database, phiên bản PostgreSQL và pgvector, sau đó `Setup OK`. Đây **chỉ là kiểm tra hạ tầng**, không phải kết quả benchmark. Docker dùng cổng `127.0.0.1:55433` và mật khẩu cố định **chỉ dùng local cho lab**; nếu cổng bận, sửa đồng thời port trong `docker-compose.yml` và `DATABASE_URL` trong `.env`. `init.sql` chỉ chạy khi volume mới được tạo; nếu database sẵn có chưa bật extension, xem log Docker và kiểm tra cấu hình thay vì xóa volume chứa dữ liệu.

Dừng container nhưng giữ dữ liệu:

```bash
docker compose -p day6_student_starter stop
```

## 2. Embedding model

Pipeline dùng model local [`Horizon-Labs/multilingual-embedding-small`](https://huggingface.co/Horizon-Labs/multilingual-embedding-small), tạo vector chuẩn hóa 1024 chiều. Không cần API key; lần chạy đầu tải weights từ Hugging Face và cần đủ RAM/VRAM. Đặt `EMBEDDING_MODEL` trong `.env` để chọn một model Sentence Transformers khác. Dùng cùng model cho toàn bộ chiến lược và vector query tương thích vector document.

## 3. Chạy pipeline

1. Đặt corpus được phép sử dụng vào `data/` và giữ nguồn/trang cùng metadata có ích cho filter.
2. Tạo benchmark JSON theo định dạng và quy tắc trong [BENCHMARK.md](../BENCHMARK.md).
3. Từ thư mục `starter/`, kích hoạt `.venv` và cài dependencies bằng `python -m pip install -r requirements.txt`.
4. Khởi động PostgreSQL + pgvector và điền `DATABASE_URL` trong `.env`.
5. Chạy `python -m src.run`. Pipeline tạo ba chiến lược A (fixed-size), B (theo paragraph), C (B + metadata filter) và lưu trace top-3, metric cùng mẫu chấm thủ công để trống vào `results/benchmark_results.json`.

Khi dùng DOCX, bộ nạp giữ dấu trang đã kết xuất trong tài liệu. Nếu benchmark evidence không khớp chính xác với text corpus, pipeline lưu cảnh báo trong kết quả để bạn kiểm tra phiên bản/trang; không tự sửa nhãn gold. Lưu code, cấu hình, kết quả và lệnh chạy để người chấm tái lập; không commit key, dữ liệu mật hoặc dữ liệu không có quyền chia sẻ.
