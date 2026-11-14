# Báo cáo cá nhân — Day 6 · Data Ingestion Engineering

**Họ và tên:** Đặng Quang Duy  
**MSSV:** 2A202603984  
**Lớp:** L3B  
**Thời điểm chạy benchmark:** 14/11/2026, 23:21 (UTC+7)

## Dữ liệu và pipeline

- **Chủ đề:** Nghị định 94/2026/NĐ-CP về hoạt động đào tạo và sát hạch lái xe.
- **Corpus thực tế:** [nghi_dinh_94_2026.docx](../../starter/data/nghi_dinh_94_2026.docx), 32 đơn vị trang có text. [manifest.json](../../data/manifest.json) liệt kê `page_01`–`page_32`, tổng 71.723 ký tự; pipeline đếm được 71.678 ký tự sau chuẩn hóa, chênh 45 ký tự cần được đối chiếu. Bộ nạp dùng 31 dấu `lastRenderedPageBreak` đã lưu trong DOCX; không có trang rỗng, không dùng OCR. Chuẩn hóa Unicode NFC, khoảng trắng và dòng nhưng giữ ranh giới đoạn.
- **Nguồn/quyền:** manifest ghi `source_path_or_url=nghi_dinh_94_2026.docx`, quyền là “Văn bản quy phạm pháp luật (Công khai / Không thuộc đối tượng bảo hộ quyền tác giả theo Luật Sở hữu trí tuệ Việt Nam)” và thời điểm truy cập `2026-11-14T22:36:00+07:00`. Snapshot revision 13; ngày ban hành 31/03/2026, `created_at=2026-03-26T04:19:00Z`, `modified_at=2026-11-14T02:50:00Z`. Manifest cung cấp đường dẫn file cục bộ, không có URL upstream. Benchmark dùng source label `nghi_dinh_94_2026_nd-cp.pdf` và khai báo một `original_file` basename khác với DOCX thực tế; pipeline dùng DOCX duy nhất trong `starter/data` và giữ source label PDF để đối chiếu gold.
- **Kiểm chứng evidence:** 24/24 quote trong benchmark được tìm thấy trong text corpus sau chuẩn hóa. Tuy nhiên, kiểm tra theo trang DOCX cục bộ cho thấy **5/24 evidence reference không nằm trọn ở trang ghi trong benchmark**: D02 (benchmark trang 2, DOCX trang 1), một evidence của D06 (trang 7 / trang 6), D12 (không nằm trọn trên một trang DOCX), T01 (trang 5 / trang 4), T02 (trang 18 / trang 17). Benchmark ghi các trang gold theo PDF; vì vậy các chỉ số cần khớp chính xác nguồn/trang có thể bị tính thiếu khi áp dụng lên ranh giới trang DOCX này.
- **Embedding:** [`Horizon-Labs/multilingual-embedding-small`](https://huggingface.co/Horizon-Labs/multilingual-embedding-small), chạy local qua Sentence Transformers 5.7.0; vector chuẩn hóa 1024 chiều, không thêm prefix query/document. Model card ghi giới hạn 512 token và giấy phép Apache-2.0 cho model. Môi trường dùng PyTorch 2.15.0 CPU (`torch.cuda.is_available() = False`), Transformers 5.23.0. Revision model chưa được pin bằng commit SHA.
- **Vector store:** PostgreSQL 16.16 + pgvector 0.8.7, bảng `ingested_chunks_1024`; exact search theo cosine distance (`<=>`), không dùng ANN index. `psycopg` 3.3.8. Không có chi phí API ở lần chạy cuối; model được tải từ Hugging Face và inference chạy local.
- **Metadata:** `document_id` (ID văn bản trong benchmark), `source` (tên PDF chuẩn), `page` (trang từ page-break DOCX), `chapter` (Chương I–V lấy từ heading), `year` (2026) và `version` (`94/2026/NĐ-CP`) lấy từ metadata benchmark. Filter sử dụng `chapter`, lấy từ phạm vi nêu rõ trong câu hỏi, không lấy từ gold source/đáp án. Trường `chapter` giúp giới hạn phạm vi văn bản; hạn chế là lọc theo chương không đảm bảo chunk trong chương đó chứa đúng bằng chứng.
- **Lệnh chạy:** từ `starter/`, kích hoạt `.venv`, cài `python -m pip install -r requirements.txt`, sau đó chạy `python -m src.run`. Kiểm tra hạ tầng bằng `python scripts/check_setup.py`.

## Cấu hình và benchmark

| Strategy | Chunking | Filter | Số chunk | Độ dài chunk (min / TB / max, ký tự) |
|---|---|---|---:|---:|
| A_fixed | Cửa sổ 800 ký tự, overlap 100 | Không | 114 | 105 / 701 / 800 |
| B_paragraph | Gom đoạn, tối đa 1.000 ký tự | Không | 99 | 112 / 723 / 1.000 |
| C_filtered | Cùng chunk và vector với B | `chapter` khi câu hỏi có filter | 99 | 112 / 723 / 1.000 |

Benchmark có 20 câu, chia 15 dev / 5 test; 3 câu dùng filter (D13, D15, T01), trong đó T01 thuộc test. Cả ba chiến lược dùng cùng corpus, model, metric và câu hỏi; B–C chỉ khác việc áp dụng filter. Tập câu hỏi và gold nằm trong [benchmark.json](../../benchmark/benchmark.json).

**Tiêu chí chọn cấu hình trước khi xem test:** không có tiêu chí chọn được ghi lại trước lần chạy này. Vì vậy kết quả dưới đây chỉ dùng để so sánh, không tuyên bố một cấu hình thắng holdout. Với lần chạy tiếp theo, có thể chốt trước tiêu chí ưu tiên Evidence Hit@3 trên dev, rồi Hit@1 và tài nguyên.

## Kết quả retrieval

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Source/page Hit@3 (n/N) | Chunk count | Thời gian embedding + upsert | Chi phí API |
|---|---|---:|---:|---:|---:|---:|---:|
| A_fixed | dev | 7/15 (46,7%) | 9/15 (60,0%) | 14/15 (93,3%) | 114 | 19,710 + 0,183 giây | 0 |
| A_fixed | test | 1/5 (20,0%) | 1/5 (20,0%) | 2/5 (40,0%) | 114 | Đã tính ở lần ingest chung | 0 |
| B_paragraph | dev | 9/15 (60,0%) | 10/15 (66,7%) | 13/15 (86,7%) | 99 | 17,873 + 0,146 giây | 0 |
| B_paragraph | test | 2/5 (40,0%) | 2/5 (40,0%) | 5/5 (100%) | 99 | Đã tính ở lần ingest chung | 0 |
| C_filtered | dev | 9/15 (60,0%) | 11/15 (73,3%) | 13/15 (86,7%) | 99 | Dùng lại B + upsert 0,153 giây | 0 |
| C_filtered | test | 2/5 (40,0%) | 2/5 (40,0%) | 5/5 (100%) | 99 | Đã tính ở lần ingest chung | 0 |

Evidence Hit yêu cầu chunk top-k khớp gold source/page và chứa đủ quote đã ghi; đây là phép kiểm tra retrieval, **không phải độ chính xác câu trả lời LLM**. Filter tuân thủ: 3/3 câu filter của C có toàn bộ top-3 đúng `chapter`, gồm 1/1 câu filter test. A/B không bật filter nên N/A.

### Thời gian và metadata

- Thời gian embedding: A 19,710 giây; B 17,873 giây. C tái sử dụng vector B nên không embedding tài liệu lần nữa.
- Thời gian upsert: A 0,183 giây; B 0,146 giây; C 0,153 giây.
- Chưa đo riêng thời gian đọc/chunk, query embedding và truy vấn search; thời gian trên không phải tổng thời gian pipeline.
- Kiểm tra tự động trên 5 câu test × top-3: `source`, `page`, `chapter`, `year`, `version` đúng với metadata trang DOCX tại **45/45 bản ghi** (15 mỗi strategy). Đây là precheck theo dữ liệu nạp, **không thay cho audit thủ công về tính hữu dụng**.
- `chapter` hữu ích để giới hạn phạm vi theo chương; hạn chế là không đủ phân biệt các điều khoản gần nhau trong cùng chương.

### Audit thủ công

File [benchmark_results.json](../../starter/results/benchmark_results.json) lưu 60 trace top-3 và mẫu audit:

| Strategy | Coherence trung bình /2 (n=10) | Metadata đủ/đúng (n=15) | Filter tuân thủ | Grounding thủ công (5 câu test) |
|---|---|---|---|---|
| A_fixed | Chưa chấm | 15/15 precheck tự động; usefulness chưa chấm | N/A | Chưa chấm; chưa có phân bố 0/1/2 |
| B_paragraph | Chưa chấm | 15/15 precheck tự động; usefulness chưa chấm | N/A | Chưa chấm; chưa có phân bố 0/1/2 |
| C_filtered | Chưa chấm | 15/15 precheck tự động; usefulness chưa chấm | 3/3 câu filter, gồm 1/1 test | Chưa chấm; chưa có phân bố 0/1/2 |

Coherence dùng 10 vị trí cố định trên 10 trang đầu có text, có chunk tương ứng cho A/B/C. Grounding có 15 mẫu (5 câu test × 3 strategy); nhãn và lý do còn trống. Không suy diễn nhãn thủ công từ Evidence Hit@3.

## Phân tích và kết luận

- **A–B:** trên dev, B tăng Hit@1 từ 7/15 lên 9/15 và Evidence Hit@3 từ 9/15 lên 10/15, đồng thời giảm số chunk từ 114 xuống 99. Trên test, B đạt 2/5 ở cả hai metric so với 1/5 của A. Đây là quan sát trên một corpus và 5 câu test, chưa đủ để suy rộng.
- **B–C:** C tăng Evidence Hit@3 dev thêm 1 câu (11/15 so với 10/15); Hit@1 dev và cả hai metric test không đổi. Filter tuân thủ đầy đủ nhưng chưa cải thiện retrieval evidence trên test; chi phí thêm là truy vấn filter, không phải embedding mới.
- **Ví dụ chưa tìm đủ evidence:** T01 hỏi về lưu lượng 500 học viên và lọc đúng `chapter_II`; C trả về các trang 4, 5, 4 nhưng không có chunk top-3 chứa đủ quote gold nên Evidence Hit@3 = 0. Đây là hạn chế tìm evidence dù filter đúng.
- **Ví dụ nhầm đoạn:** T05 hỏi quy định chuyển tiếp hồ sơ nộp trước ngày hiệu lực; B/C trả về các trang 31, 22, 20 nhưng không có chunk chứa đủ evidence gold, nên Evidence Hit@3 = 0.
- **Ảnh hưởng lệch trang:** với T02, quote gold ghi trang 18 nhưng xuất hiện trong text DOCX ở trang 17; top-3 B/C có chunk trang 17 chứa quote nhưng phép chấm nghiêm ngặt vẫn tính miss do lệch trang. Cần đồng bộ corpus PDF/DOCX hoặc sửa ánh xạ trang đã xác minh trước khi dùng các tỷ lệ này làm kết luận cuối.
- Nếu tạm xét riêng dev, C có Evidence Hit@3 cao nhất (11/15), nhưng chưa có tiêu chí chọn được đăng ký trước và test B/C bằng nhau ở 2/5. **Chưa kết luận C là cấu hình tốt nhất**; cần xử lý lệch trang và hoàn thành audit thủ công trước khi chốt.
- Kết quả cuối dùng model local. Lần thử Gemini trước đó không tạo benchmark hoàn chỉnh (gặp giới hạn quota/API) và không nằm trong kết quả này. Không có seed vì pipeline không lấy mẫu ngẫu nhiên; revision model và URL/license/access date của corpus chưa được ghi.