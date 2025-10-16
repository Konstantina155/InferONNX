#!/bin/bash
set -e

MODEL_DIR="models"

echo "Downloading model ResNet101 v2..."
wget -O $MODEL_DIR/resnet101-v2-7/resnet101-v2-7.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.onnx

echo "Downloading model ResNet152 v2..."
wget -O $MODEL_DIR/resnet152-v2-7/resnet152-v2-7.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.onnx

echo "Downloading model Efficientnet-v2..."
wget -O $MODEL_DIR/efficientnet-v2-l-18/efficientnet-v2-l-18.onnx https://huggingface.co/onnxmodelzoo/efficientnet_v2_l_Opset18/resolve/main/efficientnet_v2_l_Opset18.onnx?download=true

echo "Download complete."