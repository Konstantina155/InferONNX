import os

model_directory = ["squeezenet1.0-7", "mobilenetv2-7", "efficientnet-lite4-11", "resnet101-v2-7", "resnet152-v2-7", "densenet-7", "inception-v3-12", "efficientnet-v2-l-18"]

for model_dir in model_directory:
    os.system(f'rm -rf models/{model_dir}/operators/')