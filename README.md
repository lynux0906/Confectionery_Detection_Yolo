# **Confectionery Detection with Yolo Algorithm**
Author: Lê Văn Linh & Bùi Tất Hiệp 

*We'll build program to detect some kind of confectionery* 

- Kiến thức cơ bản: 
  - Python
  - Opencv
  - Convolutional neural network

**I. LƯU ĐỒ THUẬT TOÁN**

<p align="center">
  <img src="https://i.imgur.com/N4h5wJp.png"/>
</p>

Nhận diện vật được diễn ra với 2 quá trình
1. Training: Tất cả dữ liệu đầu vào sẽ được qua bước tiền xử lí trước khi phân loại và trích xuất đặc tính. Sau đó sẽ được đưa vào huấn luyện.
2. Detection: Ảnh đầu vào cũng sẽ được đưa qua tiền xử lí và phân loại, trích xuất đặc tinh. Sau đó ảnh sẽ được đưa qua mô hình đã được huấn luyện trước đó và đưa đến đầu ra.

**II. MÔ TẢ THUẬT TOÁN**

- YOLO (You only look once)

Yolo là một mô hình dạng CNN cho việc phát hiện, phân loại và nhận dạng đối tượng. Yolo được tạo ra từ việc kết hợp giữa các Convolutional Layers và Connected Layers. Trong đó các Convolutional Layers trích xuất ra các đặc tính của ảnh và Full-Connected Layers sẽ dự đoán xác suất và toạ độ của đối tượng. Thuật toán được sử dụng trong lĩnh vực Object Detection – Một bài toán quan trọng trong lĩnh vực Computer Vision.

<p align="center">
  <img src="https://i.imgur.com/jYWLvCC.png"/>
</p>

Đầu vào của mô hình là một ảnh, mô hình sẽ nhận dạng ảnh có đối tượng nào hay không, sau đó sẽ xác định toạ độ của đối tượng trong bức ảnh. Ảnh đầu vào được chia thành S x S ô (thường là 3 x 3, 7 x 7, 9 x 9…).

<p align="center">
  <img src="https://i.imgur.com/09jXbEO.png"/>
</p>

Với đầu vào là 1 ảnh, đầu ra của mô hình sẽ là một ma trận 3 chiều có kích thước S x S x (5 x N + M) với N và M lần lượt là số lượng Box và Class mà mỗi ô cần dự đoán. Mỗi bouding box cần dự đoán 5 thành phần: x, y, w, h và predicttion. Với (x, y) là toạ độc tâm của bounding box, (w, h) lần lượt là chiều rộng và chiều cao của bounding box.

<p align="center">
  <img src="https://i.imgur.com/Ls4nYxM.jpg"/>
</p>

Prediction là xác suất của vật được tính là:
Pr(Object)∗ IOU(pred,truth)

Trong đó: 
 - Pr (Object): điểm dự đoán vật
 - IOU (pred, truth): là tỉ lệ diện tích 2 bounding box chồng chéo nhau với diện tích tổng

<p align="center">
  <img src="https://i.imgur.com/RH3n2SM.png"/>
</p>

**III. XÂY DỰNG CHƯƠNG TRÌNH**

*1. Chuẩn bị dữ liệu*
Dữ liệu ảnh đầu vào bao gồm 3 loại nhãn hàng bánh kẹo khác nhau: 
- Chocopie: 1020 ảnh
 
<p align="center">
  <img src="https://i.imgur.com/szq4Dgt.png"/>
</p>
 
- Oishi: 1046 ảnh
 
<p align="center">
  <img src="https://i.imgur.com/FFWCE2K.png"/>
</p>
 
- Oreo: 1024 ảnh

<p align="center">
  <img src="https://i.imgur.com/G5FRS1M.png"/>
</p>

Dữ liệu hình ảnh đầu vào được chụp ngẫu nhiên với nhiều backgound khác nhau, cường độ ánh sáng khác nhau và nhiều góc cạnh khác nhau.

*2. Găn nhãn*

Gắn nhãn là 1 bước cần có trước khi training dữ liệu.

<p align="center">
  <img src="https://i.imgur.com/LBzRtVg.png"/>
</p>

Dữ liệu ảnh bánh kẹo được dán nhãn bằng Labelimage với 3 loại nhãn: “Oreo”, “Chocopie” và “Oishi”. Nhãn của vật được lưu vào file ".names"

<p align="center">
  <img src="https://i.imgur.com/3o83P23.png"/>
</p>

Quá trình dán nhãn sẽ cung cấp thông tin cần thiết của vật trong ảnh cho việc training.
 
<p align="center">
  <img src="https://i.imgur.com/Yt6eSKQ.png"/>
</p>

Khi dán nhãn, thông tin của vật được lưu vào file ".txt":
-	0: Thứ tự của nhãn 
-	(0.510013, 0.614341): Tỉ lệ tọa độ tâm của hộp giới hạn
-	(0.387597, 0.166667): Tỉ lệ chiều dài và chiều rộng của hộp giới hạn

*3. Training*
Ước tính thời gian training: 
- Với mỗi ảnh chúng ta có 1 bounding box, YOLO sẽ cần dự đoán tổng cộng 3000 bounding boxes. Giả sử mỗi batch của chúng ta có kích thước 64 ảnh và số lượng max_batches = 6000. Như vậy chúng ta cần dự đoán 1,152 tỉ bounding boxes
- Thời gian huấn luyện dự đoán là 24h.

<p align="center">
  <img src="https://i.imgur.com/abfpWy8.png"/>
</p>

Theo kết quả nhận được, đồ thị loss function cho thấy thuật toán đã hội tụ sau khoảng 1000 batches. Sai số ở giai đoạn sau có xu hướng tiệm cận 0.

<p align="center">
  <img src="https://i.imgur.com/fINcgx6.png"/>
</p>

Sau khi training xong, chúng ta thu được 1 file ".weights" để chạy chương trình nhận diện.

*4. Kết quả*

Chương trình được chạy thử nghiệm trên Laptop sử dụng camera 2D và cho ra kết quả với tỉ lệ chính xác cao. 

Sau quá trình xử lí ảnh, chương trình cho ra một kết quả nhận diện chính xác từng loại bánh kẹo, Những thông tin của vật sẽ được hiển thị trên màn hình máy tính, bao gồm:
•	Bounding box
•	Label
•	Confidence
•	Fps

*Oreo*

<p align="center">
  <img src="https://i.imgur.com/TRtXc8u.png"/>
</p>

*Oishi*

<p align="center">
  <img src="https://i.imgur.com/JwFwUbR.png"/>
</p>

*Chocopie*

<p align="center">
  <img src="https://i.imgur.com/bTZP2ew.jpg"/>
</p>

# Good luck!!!!
