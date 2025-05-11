import re
import os
import sys
import subprocess

def check_files_exist(files, extension):
    for file in files:
        if not os.path.exists(file) or not file.endswith(extension):
            print(f"Error: File {file} does not exist or does not end with '{extension}'")
            sys.exit(1)

def filter_dir(dir_path, name) -> int:
    full_path = os.path.join(dir_path, name)
    return os.path.isdir(full_path)

def check_onnx_models(path_models):
    if os.path.isfile(path_models):
        onnx_models = [path_models]
    elif os.path.isdir(path_models):
        onnx_models = [f for f in os.listdir(path_models) if not os.path.isdir(os.path.join(path_models, f))]
    else:
        raise ValueError(f"The provided path is neither a file nor a directory: {path_models}")

    # Check that all files are '.onnx'
    check_files_exist([os.path.join(path_models, onnx_model) for onnx_model in onnx_models], '.onnx')

def check_output_file(output_file):
    if not output_file.endswith('.out'):
        print(f'Error: Output file {output_file} does not end with ".out"')
        sys.exit(1)

def parse_arguments():
    output_file = None
    input_files = []

    if len(sys.argv) < 3:
        print('Usage: python3 peak_memory_tool.py -f <output_file> -m <path_to_models/model> -i <input_file1> <input_file2> ... <input_fileN>')
        sys.exit(1)

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-f':
            output_file = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            i += 2
        elif sys.argv[i] == '-m':
            path_to_models = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            i += 2
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
    check_files_exist(input_files, '.pb')

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
                number = match.group(1).replace(',', '') if ',' in match.group(1) else match.group(1)
                number = int(number) / 1048576  # Convert to MB
                third_numbers_array.append(number)
            elif line.startswith("Number of snapshots:"):
                snapshots = int(line.split(":")[1].strip())

        if snapshots != len(third_numbers_array):
            print(f"Warning: Number of snapshots ({snapshots}) does not match the number of extracted values ({len(third_numbers_array)})")
            return None

    return third_numbers_array

def run_command(command):
    try:
        output = subprocess.Popen(command, shell=True)
        output.wait() 
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise

def main():
    path_to_models, input_files, output_file = parse_arguments()

    command = f'valgrind --tool=massif --time-unit=ms --detailed-freq=1 --max-snapshots=1000 --massif-out-file={output_file} src/server_with_tls/scripts/./standalone_inference {path_to_models} ' + ' '.join(input_files)
    print(f'Running command: {command}')
    run_command(command)

    txt_file = output_file[:-4] + '.txt'
    run_command(f'ms_print {output_file} > {txt_file}')

    third_numbers = extract_third_number_from_file(txt_file)
    if third_numbers is None:
        print('Error: Could not extract memory usage')
        sys.exit(1)

    max_values = sorted(third_numbers, reverse=True)[:15]
    print("Top 20 maximum values:", max_values)
    print(f'Peak memory usage is: {max_values[0]} MB')

    run_command(f'rm -f {output_file} {txt_file}')

if __name__ == "__main__":
    main()