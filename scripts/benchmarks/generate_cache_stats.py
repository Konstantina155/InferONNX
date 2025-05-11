import re
import os
import sys
import subprocess
import pandas as pd
from statistics import mean

MODEL_DIRS = [
    "squeezenet1.0-7", "mobilenetv2-7", "densenet-7", "efficientnet-lite4-11",
    "inception-v3-12", "resnet101-v2-7", "resnet152-v2-7", "efficientnet-v2-l-18"
]

MODEL_NAMES = [
    "SqueezeNet 1.0", "MobileNet V2", "DenseNet121", "EfficientNet Lite4",
    "Inception V3", "ResNet101 V2", "ResNet152 V2", "EfficientNet V2"
]

def extract_event_count(label, text):
    match = re.search(rf"of event '{label}'\n# Event count \(approx\.\): (\d+)", text)
    if not match:
        raise RuntimeError(f"Event count for '{label}' not found.")
    return int(match.group(1))

def run_command(cmd):
    subprocess.run(cmd, check=True, shell=True)

def run_perf_report(cmd):
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (analysis, err) = result.communicate()
    if err:
        print(f"Error: {err}")
        exit(1)

    return analysis.decode('utf-8')

def collect_stats(num_runs):
    stats = {'Model': [], 'IPC': []}

    for model_dir, model_name in zip(MODEL_DIRS, MODEL_NAMES):
        cycles_list = []
        instr_list = []

        for _ in range(num_runs):
            record_cmd = f"sudo perf record -e cycles,instructions ./standalone_inference ../../../models/{model_dir}/ ../../../models/{model_dir}/test_data_set_0/input_0.pb"
            run_command(record_cmd)

            report = run_perf_report("sudo perf report --stdio")
            cycles = extract_event_count('cycles', report)
            instr = extract_event_count('instructions', report)

            cycles_list.append(cycles)
            instr_list.append(instr)

            os.remove("perf.data")

        avg_cycles = mean(cycles_list)
        avg_instr = mean(instr_list)
        ipc = avg_instr / avg_cycles if avg_cycles else 0

        stats['Model'].append(model_name)
        stats['IPC'].append(f"{ipc:.2f}")

    return stats

def generate_csv(stats):
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(stats)
    df.to_csv("results/table3.csv", index=False)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 generate_cache_stats.py <number_of_runs>")
        exit(1)

    number_of_runs = int(sys.argv[1])
    original_dir = os.getcwd()
    os.chdir(f"src/server_with_tls/scripts")
    
    try:
        run_command("make clean")
        run_command(f"make USE_CACHE_STATS={number_of_runs}")
        stats = collect_stats(number_of_runs)
        run_command("make clean")
    finally:
        os.chdir(original_dir)

    generate_csv(stats)

if __name__ == "__main__":
    main()