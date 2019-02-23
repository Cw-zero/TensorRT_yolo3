# Use TensorRT accelerate yolo3
---
## 1. How to run this project
- a. Download yolo3.weight from [this](https://pjreddie.com/media/files/yolov3.weights), and change the name to **yolov3-608.weights**.
- b. `python yolov3_to_onnx.py`, you will have a file named **yolov3-608.onnx**
- c. `python onnx_to_tensorrt.py`,you can get the result of detections.

##2. Performance compare
- a.You can download and run [this project](https://github.com/ayooshkathuria/pytorch-yolo-v3), which our project is changed from it.
It detection speed is about **100ms** per image.

- b.Our project speed is about **62ms** per image

##3.Others
- if you are Chinese people, you can also reference this blog(https://www.cnblogs.com/justcoder/).
you can read more deals.
