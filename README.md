# Update on 2019-04-19
- I have optimized and upgraded this project.So：
- If you see this project for the first time, you can jump to [This project](https://github.com/Cw-zero/TensorRT_yolo3_module) directly. 
- If you meet some bug on this project,you can try [This project](https://github.com/Cw-zero/TensorRT_yolo3_module).

# Use TensorRT accelerate yolo3
---
## 1. How to run this project
- a. Download yolo3.weight from [this](https://pjreddie.com/media/files/yolov3.weights), and change the name to **yolov3-608.weights**.
- b. `python yolov3_to_onnx.py`, you will have a file named **yolov3-608.onnx**
- c. `python onnx_to_tensorrt.py`,you can get the result of detections.

## 2. Performance compare
- a.You can download and run [this project](https://github.com/ayooshkathuria/pytorch-yolo-v3), which our project is changed from it.
It detection speed is about **100ms** per image.

- b.Our project speed is about **62ms** per image

## 3.Others
- If you are more familiar with Chinese, you can refer to this blog(https://www.cnblogs.com/justcoder/), which has more details.
