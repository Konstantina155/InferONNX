import subprocess

model_directory = [
    "gpt2", "smol-llama-220M-GQA", "mistral-300M", "qwen2.5-0.5B"
]

for model_dir in model_directory:
    subprocess.run(["rm", "-rf", f"models/{model_dir}/original_intra_ops/", f"models/{model_dir}/partitions_test/"], check=True)

subprocess.run(["rm", "-rf", "memory_intensive_ops/"], check=True)
subprocess.run(["rm", "*.csv"], check=True)