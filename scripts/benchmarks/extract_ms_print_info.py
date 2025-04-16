import re
import os
import sys
import numpy as np

if len(sys.argv) != 2 and len(sys.argv) != 1:
    print("Usage: python3 extract_ms_print_info.py <partitions_folder/''>")
    exit(1)

previous_path = os.getcwd()
prefix = f'{previous_path}/massif_output/'
if not os.path.exists(prefix):
    os.mkdir(prefix)
file_paths = ['squeezenet1.0-7.txt', 'mobilenetv2-7.txt', 'densenet-7.txt', 'efficientnet-lite4-11.txt', 'inception-v3-12.txt', 'resnet101-v2-7.txt', 'resnet152-v2-7.txt', 'efficientnet-v2-l-18.txt']
paths = ["squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", "efficientnet-lite4-11/", "inception-v3-12/", "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2-l-18/"]

partitions_folder = ""
if len(sys.argv) == 2:
    partitions_folder = sys.argv[1]
    file_paths = [path.replace('.txt', '_partitions.txt') for path in file_paths]

def generate_ms_print_info():
    id = 0
    for path in paths:
        if partitions_folder == "":
            output_file = prefix + path[:-1] + ".out"
        else:
            output_file = prefix + path[:-1] + "_partitions.out"
        command = f'valgrind --tool=massif --time-unit=ms --detailed-freq=1 --max-snapshots=1000 --massif-out-file={output_file} ./standalone_inference {previous_path}/models/{path}{partitions_folder} {previous_path}/models/{path}test_data_set_0/input_0.pb'
        print(f'Running command: {command}')
        
        os.system(command)
        txt_file = output_file[:-4] + '.txt'
        os.system(f'ms_print {output_file} > {txt_file}')

        id += 1

def extract_third_number_from_file(file_path):
    pattern = re.compile(r"\s*\d[\d,]*\s+\d[\d,]*\s+(\d[\d,]*)\s+\d[\d,]*\s+\d[\d,]*\s+\d[\d,]*")
    pattern2 = re.compile(r"^\s*(\d[\d,]*)\s+")

    third_numbers_array = []
    snapshots = 0
    flag = False
    with open(file_path, 'r') as file:
        for line in file:
            if not flag and "execute_tree" in line:
                print("Found main")
                snapshots -= snapshot_when_alloc - 1
                print(f"Snapshot when alloc: {snapshot_when_alloc}")
                third_numbers_array = [number]
                flag = True
            match = pattern.search(line)
            match2 = pattern2.search(line)
            if match2:
                snapshot_when_alloc = int(match2.group(1).replace(',', '')) + 1
            if match and not line.startswith(" Detailed snapshots:"):
                if ',' in match.group(1):
                    number = match.group(1).replace(',', '')
                else:
                    number = match.group(1)
                number = int(number) / 1048576  # Convert to MB
                third_numbers_array.append(number)
            else:
                if line.startswith("Number of snapshots:"):
                    snapshots = int(line.split(":")[1].strip())

        if snapshots != len(third_numbers_array):
            print(f"Warning: Number of snapshots ({snapshots}) does not match the number of extracted values ({len(third_numbers_array)})")
            return None

    return third_numbers_array

def extract_model(model_path):
    return model_path.rsplit('/', 1)[-1]

def analyze_memory_usage(file_paths):
    if partitions_folder != "":
        thresholds = list(range(0, 1001))
        filename = 'scripts/benchmarks/memory_requirements_detailed_partitions.txt'
    else:
        thresholds = list(range(0, 1001))
        filename = 'scripts/benchmarks/memory_requirements_detailed.txt'

    with open(filename, 'a') as f:
        for file in file_paths:
            print(f"Analyzing {file}")
            file = prefix + file
            third_numbers = extract_third_number_from_file(file)
            if third_numbers is None or len(third_numbers) == 0:
                print(f"Skipping {file}")
                continue

            counts = [0] * len(thresholds)

            for number in third_numbers:
                for i, threshold in enumerate(thresholds):
                    if number >= threshold:
                        counts[i] += 1

            f.write(f"Model: {extract_model(file)}\n")
            for i, threshold in enumerate(thresholds):
                percentage = (counts[i] / len(third_numbers)) * 100
                f.write(f"    exceeds {threshold}MB by: {percentage:.2f}%\n")
            f.write("\n")

if __name__ == "__main__":
    os.chdir(f"{previous_path}/src/server_with_tls/scripts")
    os.system("make clean && make")

    generate_ms_print_info()

    os.chdir(f"{previous_path}/src/server_with_tls/scripts")
    os.system("make clean")
    os.chdir(previous_path)

    analyze_memory_usage(file_paths)

    os.system(f"rm -rf {previous_path}/massif_output/")