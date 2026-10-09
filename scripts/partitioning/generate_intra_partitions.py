import os
import sys
import csv
import subprocess

def get_disk_size_mb(tar_path):
    if not os.path.exists(tar_path):
        return "-"
    return round(os.path.getsize(tar_path) / (1024 * 1024), 2)

def measure_execution_time(model_name, experiment_id):
    result = subprocess.run(f"python3 {inferONNX_path}/scripts/benchmarks/calculate_exec_time_of_ops.py {inferONNX_path}/ head_matmul {model_name} {experiment_id} nnef", shell=True, text=True, capture_output=True, check=True)
    print(result.stdout)
    return round(float(result.stdout.split("Average execution time:")[1].strip()) / 1000, 3)

def run_command(command):
    try:
        output = subprocess.Popen(command, shell=True)
        output.wait()

        if output.returncode != 0:
            raise subprocess.CalledProcessError(output.returncode, command)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise

def make_build():
    previous_path = os.getcwd()
    try:
        os.chdir(f"{inferONNX_path}/src/server_with_tls/scripts/")
        run_command("make clean && make")
    finally:
        os.chdir(previous_path)

def clean_build():
    previous_path = os.getcwd()
    try:
        os.chdir(f"{inferONNX_path}/src/server_with_tls/scripts/")
        run_command("make clean")
    finally:
        os.chdir(previous_path)

def convert_onnx_to_nnef(onnx_intra_path, nnef_intra_path):
    os.makedirs(nnef_intra_path, exist_ok=True)
    subprocess.run(f"rm -f {nnef_intra_path}/*.tar", shell=True, check=True)

    previous_path = os.getcwd()
    try:
        os.chdir(f"{inferONNX_path}/src/server_with_tls/scripts/")

        print(f"ONNX input : {onnx_intra_path}")
        print(f"NNEF output: {nnef_intra_path}")

        run_command(f"./onnx_to_nnef_conversion {onnx_intra_path} {nnef_intra_path}")
    finally:
        os.chdir(previous_path)

def split_and_execute(model_name, split_mode, number_splits, num_partitions, splitting_op):
    if splitting_op == "matmul":
        splitting_op_file = num_partitions - 1
    else:
        splitting_op_file = 1
    partitions_path = f"{onnx_model_path}/{model_name}/partitions_intra/"
    
    # command = f"python3 {inferONNX_path}/partitioning/split_{splitting_op}_operator.py {model_name} {split_mode} {number_splits} {inferONNX_path}/"
    # output = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    # (_, err) = output.communicate()
    # if err:
    #     print(f"Error executing command: {command}")
    #     sys.exit(1)

    subprocess.run(f"rm -f {onnx_model_path}/{model_name}/partitions_intra/{model_name}_split{splitting_op_file}_* {onnx_model_path}/{model_name}/partitions_intra/{model_name}_split{splitting_op_file}.onnx \
                    && cp {onnx_model_path}/{model_name}/partitions_test/{number_splits}_parts_{split_mode}/{model_name}_split{splitting_op_file}_* {partitions_path}", shell=True, text=True)
    result = subprocess.run(f"ls {onnx_model_path}/{model_name}/partitions_intra/ | wc -w", shell=True, text=True, capture_output=True)
    num_partitions_intra = int(result.stdout.strip())

    result = subprocess.run(f"ls {onnx_model_path}/{model_name}/partitions_test/{number_splits}_parts_{split_mode}/{model_name}_split{splitting_op_file}_* | wc -w", shell=True, text=True, capture_output=True)
    num_op_parts = int(result.stdout.strip())

    if num_partitions_intra != (num_op_parts + num_partitions - 1):
        print(num_partitions_intra, num_op_parts)
        print(f"Number of partitions + the split parts does not match the partitions intra when splitting {splitting_op}!\n")
        sys.exit(1)

    onnx_intra_path = os.path.join(onnx_model_path, model_name, "partitions_intra")
    nnef_intra_path = os.path.join(nnef_model_path, model_name, "partitions_intra")
    convert_onnx_to_nnef(onnx_intra_path, nnef_intra_path)

    output = subprocess.Popen(f"python3 {inferONNX_path}/scripts/run_all.py split_op {model_name} partitions_intra/ 1", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = output.communicate()

    print("STDOUT:")
    print(stdout.decode())

    print("STDERR:")
    print(stderr.decode())

    print("RETURN CODE:", output.returncode)
    print(output.returncode)

    if output.returncode != 0:
        return None
    return "Ok"
        
def place_optimal_split_method_into_place(model_name, split_mode, number_splits, num_partitions, splitting_op):
    if splitting_op == "matmul":
        splitting_op_file = num_partitions - 1
    else:
        splitting_op_file = 1
    partitions_path = f"{onnx_model_path}/{model_name}/partitions_intra/"

    subprocess.run(f"rm -f {onnx_model_path}/{model_name}/partitions_intra/* && cp {onnx_model_path}/{model_name}/partitions_inter/* {partitions_path}", shell=True, text=True)

    subprocess.run(f"rm -f {onnx_model_path}/{model_name}/partitions_intra/{model_name}_split{splitting_op_file}_* {onnx_model_path}/{model_name}/partitions_intra/{model_name}_split{splitting_op_file}.onnx \
                    && cp {onnx_model_path}/{model_name}/partitions_test/{number_splits}_parts_{split_mode}/{model_name}_split{splitting_op_file}_* {partitions_path}", shell=True, text=True)
    result = subprocess.run(f"ls {onnx_model_path}/{model_name}/partitions_intra/ | wc -w", shell=True, text=True, capture_output=True)
    num_partitions_intra = int(result.stdout.strip())

    result = subprocess.run(f"ls {onnx_model_path}/{model_name}/partitions_test/{number_splits}_parts_{split_mode}/{model_name}_split{splitting_op_file}_* | wc -w", shell=True, text=True, capture_output=True)
    num_op_parts = int(result.stdout.strip())

    if num_partitions_intra != (num_op_parts + num_partitions - 1):
        print(num_partitions_intra, num_op_parts)
        print(f"Number of partitions + the split parts does not match the partitions intra when splitting {splitting_op}!\n")
        sys.exit(1)

    onnx_intra_path = os.path.join(onnx_model_path, model_name, "partitions_intra")
    nnef_intra_path = os.path.join(nnef_model_path, model_name, "partitions_intra")
    os.makedirs(nnef_intra_path, exist_ok=True)
    subprocess.run(f"rm -f {nnef_intra_path}/*.tar", shell=True, check=True)

    convert_onnx_to_nnef(onnx_intra_path, nnef_intra_path)

def main():
    global onnx_model_path, nnef_model_path, inferONNX_path
    inferONNX_path = os.getcwd()
    number_of_divisions = ["2", "4", "8", "16","32", "64", "128", "256", "512"]
    split_modes = ["row", "column"]
    models = ["gpt2", "smol-llama-220M-GQA", "mistral-300M", "qwen2.5-0.5B"]
    onnx_model_path = os.path.join(inferONNX_path, "models/onnx/")
    nnef_model_path = os.path.join(inferONNX_path, "models/nnef/")

    experiments = [
        (split_mode, number_splits)
        for split_mode in split_modes
        for number_splits in number_of_divisions
    ]

    make_build()
    result_path = os.path.join(inferONNX_path, "results")
    os.makedirs(result_path, exist_ok=True)
    csv_path = os.path.join(result_path, "table2_journal.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["Model", "Strategy", "Partition count", "Disk size per partition (MB)", "TTFT (s)"])

        for model_name in models:
            result = subprocess.run(f"ls {onnx_model_path}/{model_name}/partitions_inter/ | wc -w", shell=True, text=True, capture_output=True)
            num_partitions = int(result.stdout.strip())

            partitions_path = f"{onnx_model_path}/{model_name}/partitions_intra/"
            os.makedirs(partitions_path, exist_ok=True)
            subprocess.run(f"cp {onnx_model_path}/{model_name}/partitions_inter/* {partitions_path}", shell=True, text=True)

            # ----- Experiment 1: No split -----
            if model_name == "gpt2":
                csv_model_name = "GPT-2 (0.124B)"
            elif model_name == "smol-llama-220M-GQA":
                csv_model_name = "LLaMA 2 (0.22B)"
            elif model_name == "mistral-300M":
                csv_model_name = "Mistral (0.3B)"
            elif model_name == "qwen2.5-0.5B":
                csv_model_name = "Qwen2.5 (0.5B)"
            else:
                csv_model_name = model_name
            no_split_tar = os.path.join(onnx_model_path, model_name, "partitions_inter", f"{model_name}_split{num_partitions - 1}.onnx")
            no_split_ttft = measure_execution_time(model_name, 1)
            no_split_size = get_disk_size_mb(no_split_tar)
            writer.writerow([csv_model_name, "No split", 1, no_split_size, no_split_ttft])

            # ----- Experiment 2 till ... : Split operator in Row and Column for 2^X op partitions until memory exhaustion -----
            optimal_ttft = float("inf")
            optimal_split_method = None
            optimal_number_splits = None
            for experiment_id, (split_mode, number_splits) in enumerate(experiments, start=2):
                #result = split_and_execute(model_name, split_mode, number_splits, num_partitions, "matmul")
                result = "ok"
                tar_path = os.path.join(onnx_model_path, model_name, "partitions_test", f"{number_splits}_parts_{split_mode}", f"{model_name}_split{num_partitions - 1}_0.onnx")
                disk_size = get_disk_size_mb(tar_path)
                if result is None:
                    ttft = "-"
                else:
                    try:
                        ttft = measure_execution_time(model_name, experiment_id)
                        if ttft == 0.0: ttft = '-'
                    except Exception as e:
                        ttft = '-'

                writer.writerow([csv_model_name, split_mode.capitalize(), number_splits, disk_size, ttft])
                csv_file.flush()
                print(f"Result -> partitioning scheme: {split_mode}_{number_splits}, TTFT: {ttft}s, Disk size: {disk_size}MB\n")
            
                if ttft != "-":
                    if ttft < optimal_ttft:
                        optimal_ttft = ttft
                        optimal_split_method = split_mode
                        optimal_number_splits = number_splits

            place_optimal_split_method_into_place(model_name, optimal_split_method, optimal_number_splits, num_partitions, "matmul")

    clean_build()

if __name__ == "__main__":
    main()
