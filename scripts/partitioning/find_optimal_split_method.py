import sys
import os
import subprocess

def split_and_execute(model_name, split_mode, number_splits, num_partitions, splitting_op):
    if splitting_op == "matmul":
        splitting_op_file = num_partitions - 1
    else:
        splitting_op_file = 1
    partitions_path = f"{model_path}/{model_name}/partitions_intra/"
    
    command = f"python3 {inferONNX_path}split_{splitting_op}_operator.py {model_name} {split_mode} {number_splits} {inferONNX_path}"
    output = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (_, err) = output.communicate()
    if err:
        print(f"Error executing command: {command}")
        sys.exit(1)

    subprocess.run(f"rm -f {model_path}/{model_name}/partitions_intra/{model_name}_split{splitting_op_file}_* {model_path}/{model_name}/partitions_intra/{model_name}_split{splitting_op_file}.onnx \
                    && cp {model_path}/{model_name}/partitions_test/{number_splits}_parts_{split_mode}/{model_name}_split{splitting_op_file}_* {partitions_path}", shell=True, text=True)
    result = subprocess.run(f"ls {model_path}/{model_name}/partitions_intra/ | wc -w", shell=True, text=True, capture_output=True)
    num_partitions_intra = int(result.stdout.strip())

    result = subprocess.run(f"ls {model_path}/{model_name}/partitions_test/{number_splits}_parts_{split_mode}/{model_name}_split{splitting_op_file}_* | wc -w", shell=True, text=True, capture_output=True)
    num_op_parts = int(result.stdout.strip())

    if num_partitions_intra != (num_op_parts + num_partitions - 1):
        print(num_partitions_intra, num_op_parts)
        print(f"Number of partitions + the split parts does not match the partitions intra when splitting {splitting_op}!\n")
        sys.exit(1)

    output = subprocess.Popen(f"python3 {inferONNX_path}scripts/run_all.py partitions_intra/ 1", stdout=subprocess.PIPE, shell=True)
    (_, err) = output.communicate()
    if err:
        print(f"Error executing command: {command}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("python3 calc_optimal_split_method.py <path_to_InferONNX>")
        sys.exit(1)

    # Matmul optimal sol: 128 column for both gpt2 and cerebras-gpt-111M
    global model_path, inferONNX_path
    inferONNX_path = sys.argv[1]
    number_of_divisions = ["2", "4", "8", "16", "32", "64", "128", "256"]
    split_modes = ["row", "column"]
    models = ["cerebras-gpt-111M", "gpt2", "smol-llama-220M-GQA", "mistral-300M", "teeny-tiny-llama-460M", "qwen2.5-0.5B"]
    model_path = os.path.join(inferONNX_path, "models")
    for model_name in models:
        result = subprocess.run(f"ls {model_path}/{model_name}/partitions_inter/ | wc -w", shell=True, text=True, capture_output=True)
        num_partitions = int(result.stdout.strip())

        partitions_path = f"{model_path}/{model_name}/partitions_intra/"
        os.makedirs(partitions_path, exist_ok=True)
        subprocess.run(f"cp {model_path}/{model_name}/partitions_inter/* {partitions_path}", shell=True, text=True)

        for split_mode in split_modes:
            for number_splits in number_of_divisions:
                split_and_execute(model_name, split_mode, number_splits, num_partitions, "matmul")

        for i in range(2, 2 + len(number_of_divisions) * len(split_modes) + 1):
            result = subprocess.run(f"python3 {inferONNX_path}calculate_exec_time_of_ops.py head_matmul {model_name} {i}", shell=True, text=True, capture_output=True)
            print(result.stdout)
            average_time = round(float(result.stdout.split("Average execution time:")[1].strip()))
            print(average_time)

if __name__ == "__main__":
    main()
