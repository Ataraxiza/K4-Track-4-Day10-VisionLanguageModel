Họ tên: Nguyễn Tuấn Khanh
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0.0735 | [0.0494, 0.1040] | 1.074 | 0.1548 | 0.0190 | 0.1313 |
| E1 | LoRA LLM | 33,030,144 | 5.868 | 46.03 | 0.2465 | [0.2122, 0.2828] | 1.534 | 0.1306 | 0.1926 | 0.5039 |
| E2 | + projector | 87,609,344 | 6.074 | 45.45 | 0.0907 | [0.0673, 0.1167] | 1.916 | 0.1268 | 0.0282 | 0.2881 |
| E3 | + LoRA ViT | 93,900,800 | 6.309 | 54.32 | 0.0633 | [0.0448, 0.0822] | 2.357 | 0.0745 | 0.0498 | 0.1036 |
| E4 | LoRA LLM, rank 4/64 (dự kiến) | — | — | — | — | — | — | — | — | — |

Ghi chú: các số lấy từ `report/evidence/e0.json`–`e3.json`, cùng `dataset_version` `d10-21e4070301`; cả bốn evidence đều chấm đủ 311 ảnh (`missing_predictions` = 0). `parse_valid_rate` lần lượt là E0 0.9164, E1 0.9904, E2 0.8875, E3 0.8585. Tham số huấn luyện, VRAM và thời gian của E1–E3 lấy từ metadata trong evidence; E0 là zero-shot. E4 chưa có kết quả/evidence.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Theo ước lượng kiến trúc trong GUIDE: LLM cùng embedding khoảng 4.0B tham số; ViT khoảng 415M, trong đó merger/projector khoảng 109M (merger khoảng 27M và deepstack mergers khoảng 82M); toàn mô hình khoảng 4.44B. E1 cập nhật LoRA của LLM; E2 cập nhật cùng LoRA LLM và projector; E3 cập nhật thêm LoRA trên ViT. Log thực tế ghi lần lượt 33,030,144, 87,609,344 và 93,900,800 tham số huấn luyện, với 252, 252 và 348 module LoRA mục tiêu. E2 − E1 là 54,579,200 tham số, thấp hơn ước lượng tăng khoảng 109M trong GUIDE; đây là sai khác giữa ước lượng và số trainable thực tế, không nên trình bày chúng như cùng một phép đo.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate` = 0.9164 (91.64%) và không thiếu dự đoán nào trong 311 ảnh. Evidence ghi nhận 334 false positive và 548 false negative (FP/ảnh = 1.074); crack là lớp yếu nhất với F1 = 0.0190. Đây cho thấy nhiều dự đoán nhầm và bỏ sót, nhưng số liệu tổng hợp không phân biệt lỗi định vị với nhầm lớp. Evidence JSON không lưu raw output, nên tôi không trích một dòng JSON thô chưa được xác minh. Sáu hình minh hoạ E0 bên dưới cho thấy các ví dụ dự đoán/ground truth; không nên suy rộng các hình này thành tần suất lỗi trên toàn bộ tập.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Mô hình sử dụng hệ tọa độ chuẩn hóa norm1000, trong đó mỗi tọa độ nằm trong khoảng 0–1000, với 0 là mép trái/trên và 1000 là mép phải/dưới ảnh. Khi đánh giá, nếu tọa độ pixel bị hiểu nhầm là norm1000 hoặc ngược lại, bounding box dự đoán và ground truth sẽ không cùng hệ quy chiếu, khiến IoU giảm mạnh và F1 có thể giảm đáng kể, thậm chí gần 0. Ngược lại, nếu chuyển đổi nhất quán cả dự đoán và ground truth giữa hai hệ tọa độ, IoU về lý thuyết sẽ được bảo toàn, ngoại trừ sai số nhỏ do làm tròn.

Tôi đã kiểm chứng quy trình chuyển đổi bằng code Python với các bước:
- Chuyển nhãn YOLO [xc, yc, w, h] sang [x1, y1, x2, y2] trên thang 0–1000 theo công thức nhân tọa độ chuẩn hóa với 1000; làm tròn và giới hạn tọa độ trong khoảng [0, 1000].
- Tạo JSON theo định dạng {"label": ..., "bbox_2d": [x1, y1, x2, y2]}, sắp xếp box theo (y1, x1) và sử dụng prompt quy định rõ ba lớp crack, manhole, pothole cùng thang tọa độ 0–1000.
- Chạy hàm verify_box_round_trip() để quy đổi tọa độ từ norm1000 về pixel bằng công thức x_pixel = x_norm * W / 1000 và y_pixel = y_norm * H / 1000, sau đó vẽ bounding box lên ảnh gốc. Kết quả chạy trả về 9 box của ảnh mẫu, xác nhận code chuyển đổi, tạo JSON và vẽ box đã được thực thi.

Kết quả: Code đã thực hiện việc chuyển nhãn YOLO sang norm1000 và quy đổi ngược về pixel để trực quan hóa. Kết quả trực quan từ hàm verify_box_round_trip(splits["train"][0]) cho thấy các bounding box sau khi chuyển nhãn YOLO sang norm1000 và quy đổi ngược về pixel được vẽ đúng vị trí trên ảnh gốc. Điều này xác nhận quy trình chuyển đổi tọa độ hoạt động đúng về mặt trực quan. Nếu sử dụng sai hệ tọa độ khi đánh giá, IoU có thể giảm mạnh, dẫn đến F1 giảm đáng kể, thậm chí gần 0. Tuy nhiên, thí nghiệm chưa so sánh F1 giữa hai cách chấm đúng và sai hệ tọa độ, nên chưa có số liệu định lượng để xác nhận mức độ ảnh hưởng này.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Prompt và token ảnh là đầu vào điều kiện, không phải nội dung cần model sinh. Đoạn mã gán `-100` cho labels của toàn bộ prefix prompt, nên loss chỉ học câu trả lời JSON (và token kết thúc). Nếu tính loss trên prompt/token ảnh, model bị phạt vì không tái tạo đầu vào thay vì học định vị; các token đầu vào rất dài còn lấn át loss của JSON ngắn, làm tín hiệu giám sát yếu và có thể khiến model học thuộc prompt/cấu trúc thay vì dự đoán box.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Trọng số 4-bit được lưu ở dạng lượng tử và thường được giữ đóng băng; chúng không thể được cập nhật như trọng số dense thông thường để fine-tune đầy đủ projector. Cấu hình tải model bỏ qua lượng tử hoá cho `visual` và `lm_head`, nên nhánh ViT/projector vẫn ở floating point; ở E2/E3, các tham số merger/deepstack projector được chuyển sang FP32 và bật gradient. FP32 được dùng để giảm sai số làm tròn, underflow/overflow và giữ ổn định cập nhật cho projector đang huấn luyện; các trọng số nền còn lại có thể giữ FP16 để tiết kiệm VRAM.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các frame gần nhau trong cùng phiên thường gần như trùng cảnh. Chia ngẫu nhiên theo ảnh có thể đặt frame liền kề ở train và test, khiến test không độc lập và F1 cao giả; gom theo session giữ toàn bộ phiên trong đúng một split. Từ raw ảnh/nhãn, phép chia tái tạo cho đúng `dataset_version` `d10-21e4070301` (seed `day10`, salt 0):

| Split | Ảnh | Pothole (box) | Crack (box) | Manhole (box) |
|---|---:|---:|---:|---:|
| Train | 1,422 | 1,074 | 1,750 | 691 |
| Val | 276 | 66 | 470 | 103 |
| Test | 311 | 121 | 299 | 163 |

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 đạt F1 = 0.2465 (CI95 [0.2122, 0.2828]), cao nhất. E2 giảm xuống 0.0907 ([0.0673, 0.1167]) với 6.074 GB và 45.45 phút; E3 đạt 0.0633 ([0.0448, 0.0822]) với 6.309 GB và 54.32 phút. So với E1, E2 tốn thêm 0.206 GB nhưng có thời gian tương đương và F1 thấp hơn; E3 tốn thêm 0.441 GB, thêm 8.29 phút và F1 cũng thấp hơn đáng kể. Theo các kết quả hiện có, việc mở projector/LoRA ViT không đem lại lợi ích; cả ba setup đều được chấm đủ 311 ảnh, nên so sánh này không còn bị giới hạn bởi dự đoán bị thiếu.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 của F1 là E0 [0.0494, 0.1040] (độ rộng 0.0547), E1 [0.2122, 0.2828] (0.0705), E2 [0.0673, 0.1167] (0.0494), E3 [0.0448, 0.0822] (0.0374). CI trong evidence được bootstrap 1,000 lần theo ảnh. CI của E1 tách khỏi CI của E0, E2 và E3; CI E0/E2/E3 có phần chồng lấp. Điều này ủng hộ kết quả E1 tốt hơn, nhưng so sánh CI riêng không thay thế kiểm định cặp; nên bootstrap chênh lệch F1 trên cùng ảnh test để đánh giá trực tiếp. Evidence bao phủ đầy đủ 311 ảnh, nhưng CI vẫn thể hiện độ bất định khi khái quát từ tập test hữu hạn.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Không có một lớp kém nhất ở mọi setup: crack thấp nhất ở E0 (F1 = 0.0190), E2 (0.0282) và E3 (0.0498); pothole thấp nhất ở E1 (0.1306). Pothole có F1 cao nhất ở E0 (0.1548), còn manhole cao nhất ở E1/E2/E3 (0.5039, 0.2881, 0.1036). Test có 121 box pothole, 299 crack và 163 manhole. Crack có số box nhiều nhất nhưng cũng có nhiều false positive nhất trong mọi setup (E0 221, E1 339, E2 539, E3 524); số lượng mẫu đơn thuần không giải thích được hiệu năng. Kết quả cho thấy vấn đề precision/định vị đáng kể, nhưng evidence tổng hợp không xác định nguyên nhân ngữ nghĩa hay kích thước box cụ thể. Sáu ví dụ E0 dưới đây dùng GT màu xanh lá và dự đoán màu đỏ:

<p><img src="figures/e0_example_1_vlcsnap-2025-02-18-23h13m50s113.jpg" width="32%" alt="E0 matched example"><img src="figures/e0_example_2_vlcsnap-2025-02-18-23h08m10s837.jpg" width="32%" alt="E0 duplicate box example"><img src="figures/e0_example_3_vlcsnap-00204.jpg" width="32%" alt="E0 oversized box example"></p>
<p><img src="figures/e0_example_4_vlcsnap-00200.jpg" width="32%" alt="E0 false positive example"><img src="figures/e0_example_5_vlcsnap-00201.jpg" width="32%" alt="E0 missed detection example"><img src="figures/e0_example_6_vlcsnap-00208.jpg" width="32%" alt="E0 mixed error example"></p>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Notebook cấu hình E4 để thay đổi LoRA rank, so sánh rank = 4 và rank = 64 trên baseline E1, với seed 42; alpha được đặt bằng 2 × rank, tương ứng 8 và 128. Tuy nhiên, notebook không tạo được kết quả E4 có thể chấm: không có evidence `e4_rank_4.json`/`e4_rank_64.json`, và artifact `work/e4_rank_4/run.json` bị thiếu khi cell xử lý E4 được chạy.

Do đó chưa có F1/CI95 cho hai rank và không thể kết luận rank nào tốt hơn E1. Cần hoàn tất cả hai lần chạy, xác nhận đủ 311 dự đoán cho mỗi lần và xuất evidence trước khi so sánh.

## Nhìn lại
Qua các thí nghiệm E0–E3, E1 đạt F1 = 0.2465, cao hơn rõ rệt các cấu hình còn lại trên cùng 311 ảnh test; CI95 của E1 không chồng lấp với CI của E0, E2 hoặc E3. Dù vậy, hiệu năng tuyệt đối vẫn còn hạn chế (E1 có 477 false positive và 434 false negative), nên cần xem xét cả precision/recall và lỗi theo lớp, không chỉ F1. `parse_valid_rate` cao và không có dự đoán thiếu ở các evidence mới giúp tách rõ hơn lỗi định dạng khỏi lỗi định vị/phân loại.

Trong quá trình triển khai E2 và E3, tôi gặp lỗi không tương thích dtype giữa FP16 và FP32 khi huấn luyện projector và LoRA trên nhánh ViT. Các tham số huấn luyện cần giữ FP32, trong khi một số gradient sinh ra dưới dạng FP16 trong quá trình mixed-precision training, gây lỗi khi cập nhật trọng số. Tôi đã bổ sung bước kiểm tra dtype và chuyển gradient về FP32 trước khi unscale và cập nhật optimizer.

Bài học rút ra: cần kiểm tra kỹ dtype của tham số và gradient khi kết hợp FP16 với FP32, đặc biệt ở projector và ViT; lưu đủ artifact của mỗi lần chạy; đồng thời xác minh evidence bao phủ toàn bộ ảnh test và tính so sánh thống kê phù hợp trước khi kết luận hiệu quả của mô hình.