import re
import os
import sys
import subprocess
import numpy as np
    

def generate_ms_print_info(previous_path, paths, partitions_folder, prefix):
    for path in paths:
        output_file = f'{prefix}{path[:-1]}{"_partitions" if partitions_folder else ""}.out'
        
        command = [
            'valgrind', '--tool=massif', '--time-unit=ms', '--detailed-freq=1', '--max-snapshots=1000',
            f'--massif-out-file={output_file}', './standalone_inference', 
            f'{previous_path}/models/{path}{partitions_folder}', f'{previous_path}/models/{path}test_data_set_0/input_0.pb'
        ]
        print(f'Running command: {" ".join(command)}')

        subprocess.run(command, check=True)
        txt_file = output_file[:-4] + '.txt'
        subprocess.run(['ms_print', output_file], stdout=open(txt_file, 'w'))

def extract_third_number_from_file(file_path):
    pattern = re.compile(r"\s*\d[\d,]*\s+\d[\d,]*\s+(\d[\d,]*)\s+\d[\d,]*\s+\d[\d,]*\s+\d[\d,]*")
    pattern2 = re.compile(r"^\s*(\d[\d,]*)\s+")

    third_numbers_array = []
    snapshots = 0
    snapshot_when_alloc = None
    with open(file_path, 'r') as file:
        for line in file:
            if snapshot_when_alloc is None and "execute_tree" in line:
                print("Found main")
                snapshots -= snapshot_when_alloc - 1
                print(f"Snapshot when alloc: {snapshot_when_alloc}")
                third_numbers_array = [number]
            match = pattern.search(line)
            match2 = pattern2.search(line)
            if match2:
                snapshot_when_alloc = int(match2.group(1).replace(',', '')) + 1
            if match and not line.startswith(" Detailed snapshots:"):
                number = int(match.group(1).replace(',', '')) / 1048576  # Convert to MB
                third_numbers_array.append(number)
            if line.startswith("Number of snapshots:"):
                    snapshots = int(line.split(":")[1].strip())

        if snapshots != len(third_numbers_array):
            print(f"Warning: Number of snapshots ({snapshots}) does not match the number of extracted values ({len(third_numbers_array)})")
            return None

    return third_numbers_array

def analyze_memory_usage(file_paths, partitions_folder, prefix, previous_path):
    thresholds = list(range(0, 1001))
    filename = f'{previous_path}/scripts/benchmarks/memory_requirements_detailed_partitions.txt' if partitions_folder else f'{previous_path}/scripts/benchmarks/memory_requirements_detailed.txt'

    with open(filename, 'a') as f:
        for file in file_paths:
            file_path = prefix + file
            print(f"Analyzing {file_path}")
            third_numbers = extract_third_number_from_file(file_path)
            if third_numbers is None or not third_numbers:
                print(f"Skipping {file_path}")
                continue

            counts = [0] * len(thresholds)
            for number in third_numbers:
                for i, threshold in enumerate(thresholds):
                    if number >= threshold:
                        counts[i] += 1

            f.write(f"Model: {file.rsplit('/', 1)[-1]}\n")
            for i, threshold in enumerate(thresholds):
                percentage = (counts[i] / len(third_numbers)) * 100
                f.write(f"    exceeds {threshold}MB by: {percentage:.2f}%\n")
            f.write("\n")

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 1:
        print("Usage: python3 extract_ms_print_info.py <partitions_folder/''>")
        exit(1)

    previous_path = os.getcwd()
    prefix = f'{previous_path}/massif_output/'
    os.makedirs(prefix, exist_ok=True)

    partitions_folder = sys.argv[1] if len(sys.argv) == 2 else ""
    file_paths = ['squeezenet1.0-7.txt', 'mobilenetv2-7.txt', 'densenet-7.txt', 'efficientnet-lite4-11.txt', 'inception-v3-12.txt', 'resnet101-v2-7.txt', 'resnet152-v2-7.txt', 'efficientnet-v2-l-18.txt']
    paths = ["squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", "efficientnet-lite4-11/", "inception-v3-12/", "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2-l-18/"]
    file_paths = [path.replace('.txt', '_partitions.txt') for path in file_paths] if len(sys.argv) == 2 else file_paths

    os.chdir(f"{previous_path}/src/server_with_tls/scripts")
    subprocess.run("make clean && make", shell=True)

    generate_ms_print_info(previous_path, paths, partitions_folder, prefix)
    analyze_memory_usage(file_paths, partitions_folder, prefix, previous_path)

    subprocess.run(f"rm -rf {prefix}", shell=True)

    os.chdir(previous_path)

if __name__ == "__main__":
    main()