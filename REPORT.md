Họ tên: Nguyễn Tuấn Khanh
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0.0149 | [0.0031, 0.0289] | 0.264 | 0.0000 | 0.0162 | 0.0230 |
| E1 | LoRA LLM | 33,030,144 | 5.868 | 48.29 | 0.0215 | [0.0078, 0.0407] | 0.495 | 0.0147 | 0.0094 | 0.0549 |
| E2 | + projector | 87,609,344 | 6.074 | 47.73 | 0.0082 | [0.0000, 0.0192] | 0.457 | 0.0000 | 0.0046 | 0.0238 |
| E3 | + LoRA ViT | 93,900,800 | 6.309 | 57.80 | 0.0025 | [0.0000, 0.0082] | 0.669 | 0.0000 | 0.0039 | 0.0000 |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

Ghi chú: các số lấy từ `report/evidence/e0.json`–`e3.json`; mỗi file có 311 ảnh test nhưng `missing_predictions` = 233. Vì vậy các F1 này phản ánh bộ dự đoán hiện có nhưng chưa phải đánh giá đầy đủ trên test. `parse_valid_rate` lần lượt là E0 0.2251, E1 0.2476, E2 0.1994, E3 0.1929. Riêng E1, `work/e1/run.json` ghi VRAM 7.932 GB và 48.62 phút, khác metadata trong evidence (5.868 GB và 48.29 phút); cần xác minh hai artifact thuộc cùng lần chạy trước khi nộp.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Theo ước lượng kiến trúc trong GUIDE: LLM cùng embedding khoảng 4.0B tham số; ViT khoảng 415M, trong đó merger/projector khoảng 109M (merger khoảng 27M và deepstack mergers khoảng 82M); toàn mô hình khoảng 4.44B. E1 cập nhật LoRA của LLM; E2 cập nhật cùng LoRA LLM và projector; E3 cập nhật thêm LoRA trên ViT. Log thực tế ghi lần lượt 33,030,144, 87,609,344 và 93,900,800 tham số huấn luyện, với 252, 252 và 348 module LoRA mục tiêu. E2 − E1 là 54,579,200 tham số, thấp hơn ước lượng tăng khoảng 109M trong GUIDE; đây là sai khác giữa ước lượng và số trainable thực tế, không nên trình bày chúng như cùng một phép đo.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate` = 0.2251 (22.51%); đồng thời thiếu dự đoán ở 233/311 ảnh. Các lỗi nhìn thấy trong `work/e0.jsonl` là box crack quá rộng/sai vị trí và dự đoán trùng lặp. Ví dụ output thô hợp lệ về cú pháp nhưng box gần phủ hết chiều ngang ảnh: `{"image_id":"vlcsnap-00204","raw":"[{\"label\":\"crack\",\"bbox_2d\":[12,650,1000,700]}]"}`. Một ví dụ trùng lặp là ảnh `vlcsnap-2025-02-19-17h05m06s690`, nơi cùng box pothole `[112,624,157,657]` được trả về hai lần. Vì tỷ lệ thiếu dự đoán cao, đây là các lỗi tiêu biểu quan sát được, không phải thống kê tần suất trên toàn bộ 311 ảnh.

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
Trong các lần chạy được chấm, E1 tốt nhất nhưng F1 chỉ 0.0215 (CI95 [0.0078, 0.0407]). E2 giảm xuống 0.0082 ([0.0000, 0.0192]) với 6.074 GB và 47.73 phút; E3 giảm tiếp xuống 0.0025 ([0.0000, 0.0082]) với 6.309 GB và 57.80 phút. So với E1, E2 không tăng thời gian đáng kể nhưng tốn thêm khoảng 0.206 GB và F1 thấp hơn; E3 tốn thêm khoảng 0.441 GB và 9.51 phút nhưng F1 thấp hơn nhiều. Dựa trên các số này, mở projector/LoRA ViT chưa đáng; tuy nhiên, 233 dự đoán bị thiếu ở mỗi evidence khiến kết luận về chất lượng mô hình chưa đáng tin cho đến khi chạy lại đầy đủ.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 của F1 là E0 [0.0031, 0.0289] (độ rộng 0.0258), E1 [0.0078, 0.0407] (0.0329), E2 [0.0000, 0.0192] (0.0192), E3 [0.0000, 0.0082] (0.0082). Các khoảng chồng lấp; do đó chưa có cơ sở kết luận chắc chắn setup nào hơn setup khác chỉ từ các F1 điểm. CI trong evidence là bootstrap 1,000 lần theo ảnh; để so sánh cặp chặt chẽ hơn cần bootstrap chênh lệch F1 theo cùng các ảnh. Hơn nữa, thiếu 233 dự đoán mỗi lần chạy làm giảm mạnh giá trị kết luận từ CI hiện tại.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Không có một lớp kém nhất ổn định ở mọi setup: pothole có F1 = 0 ở E0/E2/E3; crack thấp nhất ở E1/E2/E3 nếu xét lớp có F1 khác 0 (E1 0.0094, E2 0.0046, E3 0.0039); manhole là lớp tốt nhất của E0/E1/E2 nhưng E3 không có true positive. Test có 121 box pothole, 299 crack và 163 manhole; crack có nhiều box nhất nhưng vẫn có 126 false positive ở E1, cho thấy số mẫu đơn thuần không giải thích được lỗi. Các output E0 có box crack quá rộng và box lặp, phù hợp với lỗi định vị/đếm; crack mảnh và dài cũng nhạy với ngưỡng IoU 0.5. Evidence chưa phân loại được nhầm lẫn ngữ nghĩa giữa các lớp. Sáu ví dụ E0 dưới đây dùng GT màu xanh lá và dự đoán màu đỏ:

<p><img src="figures/e0_example_1_vlcsnap-2025-02-18-23h13m50s113.jpg" width="32%" alt="E0 matched example"><img src="figures/e0_example_2_vlcsnap-2025-02-18-23h08m10s837.jpg" width="32%" alt="E0 duplicate box example"><img src="figures/e0_example_3_vlcsnap-00204.jpg" width="32%" alt="E0 oversized box example"></p>
<p><img src="figures/e0_example_4_vlcsnap-00200.jpg" width="32%" alt="E0 false positive example"><img src="figures/e0_example_5_vlcsnap-00201.jpg" width="32%" alt="E0 missed detection example"><img src="figures/e0_example_6_vlcsnap-00208.jpg" width="32%" alt="E0 mixed error example"></p>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Tôi chọn đổi biến LoRA rank, so sánh hai giá trị rank = 4 và rank = 64, sử dụng setup E1 làm baseline và seed 42. Theo thiết kế, LoRA alpha cần bằng 2 × rank, tương ứng alpha = 8 và alpha = 128.

Tuy nhiên, thí nghiệm E4 chưa chạy thành công nên chưa tạo được hai file evidence e4_rank_4.json và e4_rank_64.json.

Vì chưa có kết quả F1 và CI95 của E4 nên chưa thể kết luận rank 4 hay rank 64 tốt hơn, cũng chưa thể đánh giá mức cải thiện so với E1. Độ tin cậy của đánh giá hiện tại còn hạn chế vì tập test chỉ có 311 ảnh và evidence E0–E3 đều thiếu dự đoán ở 233 ảnh mỗi lần chạy. Cần hoàn thành cả hai lần chạy với cùng tập test, kiểm tra số dự đoán hợp lệ và tính CI95 để có thể đưa ra kết luận.

## Nhìn lại
Qua các thí nghiệm E0–E3, có thể thấy F1 của mô hình còn thấp và tỷ lệ dự đoán thiếu khá cao: E1 đạt F1 = 0.0215, cao nhất trong các cấu hình được đánh giá, nhưng chưa đủ cơ sở để khẳng định đây là cấu hình tốt nhất một cách chắc chắn vì các khoảng tin cậy CI95 chồng lấp và evidence còn thiếu dự đoán. Một vấn đề quan trọng là phân biệt lỗi của mô hình với lỗi trong quá trình sinh output, phân tích JSON và chấm bounding box.

Trong quá trình triển khai E2 và E3, tôi gặp lỗi không tương thích dtype giữa FP16 và FP32 khi huấn luyện projector và LoRA trên nhánh ViT. Các tham số huấn luyện cần giữ FP32, trong khi một số gradient sinh ra dưới dạng FP16 trong quá trình mixed-precision training, gây lỗi khi cập nhật trọng số. Tôi đã bổ sung bước kiểm tra dtype và chuyển gradient về FP32 trước khi unscale và cập nhật optimizer.

Bài học rút ra: cần kiểm tra kỹ dtype của tham số và gradient khi kết hợp FP16 với FP32, đặc biệt ở projector và ViT; đồng thời xác minh log huấn luyện và đánh giá trên đầy đủ ảnh test trước khi kết luận hiệu quả của mô hình.