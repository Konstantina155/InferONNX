import subprocess

model_directory = [
    "squeezenet1.0-7", "mobilenetv2-7", "efficientnet-lite4-11",
    "resnet101-v2-7", "resnet152-v2-7", "densenet-7",
    "inception-v3-12", "efficientnet-v2-l-18"
]

for model_dir in model_directory:
    subprocess.run(["rm", "-rf", f"models/{model_dir}/operators/"], check=True)

subprocess.run(["rm", "-rf", "memory_intensive_ops/"], check=True)