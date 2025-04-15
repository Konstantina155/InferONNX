import os
import sys
import re
import sys
from collections import Counter

def check_input_files(input_files):
    for input_file in input_files:
        if not os.path.exists(input_file) or not input_file.endswith('.pb'):
            print(f'Error: Input file {input_file} does not exist or does not end with ".pb"')
            sys.exit(1)

def filter_dir(dir_path, name) -> int:
    full_path = os.path.join(dir_path, name)

    try:
        return os.path.isdir(full_path)
    except OSError as e:
        print(f"stat error: {e}")
        return 0

def check_onnx_models(path_models):
    onnx_models = [f for f in os.listdir(path_models) if not filter_dir(path_models, f)]
    for onnx_model in onnx_models:
        onnx_model = os.path.join(path_models, onnx_model)
        if not os.path.exists(onnx_model) or not onnx_model.endswith('.onnx'):
            print(f'Error: ONNX model {onnx_model} does not exist or does not end with ".onnx"')
            sys.exit(1)

def check_output_file(output_file):
    if not output_file.endswith('.out'):
        print(f'Error: Output file {output_file} does not end with ".out"')
        sys.exit(1)

def check_prompt():
    output_file = None
    input_files = []

    if len(sys.argv) < 3:
        print('Usage: python3 peak_memory_tool.py -f <output_file> -m <path_to_models> -i <input_file1> <input_file2> ... <input_fileN>')
        sys.exit(1)

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-f':
            if i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
            else:
                print('Error: Missing output file after -f')
                sys.exit(1)
        elif sys.argv[i] == '-m':
            i += 2
            path_to_models = sys.argv[i - 1]
        elif sys.argv[i] == '-i':
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('-'):
                if sys.argv[i].endswith('.pb'):
                    input_files.append(sys.argv[i])
                else:
                    print(f'Error: Invalid input file: {sys.argv[i]}')
                    sys.exit(1)
                i += 1
        else:
            print(f'Error: Unknown argument {sys.argv[i]}')
            sys.exit(1)

    if not output_file:
        print('Error: Output file not specified')
        sys.exit(1)
    check_output_file(output_file)

    if not path_to_models:
        print('Error: No path_to_models specified')
        sys.exit(1)
    check_onnx_models(path_to_models)

    if not input_files:
        print('Error: No input files specified')
        sys.exit(1)
    check_input_files(input_files)
    return path_to_models, input_files, output_file

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

path_to_models, input_files, output_file = check_prompt()

command = f'valgrind --tool=massif --time-unit=ms --detailed-freq=1 --max-snapshots=1000 --massif-out-file={output_file} src/server_with_tls/scripts/./standalone_inference {path_to_models} '
command += f'{" ".join(input_files)}'

print(f'Running command: {command}')
os.system(command)
txt_file = output_file[:-4] + '.txt'
os.system(f'ms_print {output_file} > {txt_file}')

third_numbers = extract_third_number_from_file(txt_file)
if third_numbers is None:
    print('Error: Could not extract memory usage')
    sys.exit(1)

max_values = sorted(third_numbers, reverse=True)[:15]
print("Top 20 maximum values:", max_values)
print(f'Peak memory usage is: {max_values[0]} MB')

os.system(f'rm -f {output_file} {txt_file}')