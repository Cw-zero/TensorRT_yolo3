from __future__ import print_function

import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
import time
from util import *

from data_processing import PreprocessYOLO

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():

    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'yolov3-608.onnx'
    engine_file_path = "yolov3-608.trt"
    input_image_path = "./images/b.jpg"

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (608, 608)

    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)

    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)

    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
    # output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26), (1, 255, 52, 52)]

    # Do inference with TensorRT
    trt_outputs = []
    a = torch.cuda.FloatTensor()
    average_inference_time = 0
    average_yolo_time = 0
    counter = 10
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        while counter:
            # Do inference
            print('Running inference on image {}...'.format(input_image_path))
            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inference_start = time.time()
            inputs[0].host = image
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            inference_end = time.time()
            inference_time = inference_end-inference_start
            average_inference_time = average_inference_time + inference_time
            print('inference time : %f' % (inference_end-inference_start))

            # Do yolo_layer with pytorch
            inp_dim = 608
            num_classes = 80
            CUDA = True
            yolo_anchors = [[(116, 90), (156, 198), (373, 326)],
                            [(30, 61),  (62, 45),   (59, 119)],
                            [(10, 13),  (16, 30),   (33, 23)]]
            write = 0
            yolo_start = time.time()
            for output, shape, anchors in zip(trt_outputs, output_shapes, yolo_anchors):
                output = output.reshape(shape) 
                trt_output = torch.from_numpy(output).cuda()
                trt_output = trt_output.data
                trt_output = predict_transform(trt_output, inp_dim, anchors, num_classes, CUDA)

                if type(trt_output) == int:
                    continue

                if not write:
                    detections = trt_output
                    write = 1

                else:
                    detections = torch.cat((detections, trt_output), 1)
            dets = dynamic_write_results(detections, 0.5, num_classes, nms=True, nms_conf=0.45) #0.008
            yolo_end = time.time()
            yolo_time = yolo_end-yolo_start
            average_yolo_time = average_yolo_time + yolo_time
            print('yolo time : %f' % (yolo_end-yolo_start))
            print('all time : %f' % (yolo_end-inference_start))
            counter = counter -1

        average_yolo_time = average_yolo_time/10
        average_inference_time = average_inference_time/10
        print("--------------------------------------------------------")
        print('average yolo time : %f' % (average_yolo_time))
        print('average inference time : %f' % (average_inference_time))
        print("--------------------------------------------------------")

if __name__ == '__main__':
    main()
