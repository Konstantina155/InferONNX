import subprocess
import os
import threading
import json
import time as tme
import sys
import re

def run_command(cmd, cwd=None):
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)

def run_command_with_output(cmd, cwd=None):
    output = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    _, result_stderr = output.communicate()
    return result_stderr.decode('utf-8', errors='replace')

def run_command_without_output(cmd, cwd=None):
    process = subprocess.Popen(cmd, cwd=cwd, shell=True)
    process.wait()

def init_client(use_sys_time, num_tokens):
    use_sys_time_operators = 1 if configuration == "memory_only_operators" else 0
    use_memory_only = 0 if "on_disk" in configuration else 1
    use_file_caching = 1 if configuration == "on_disk_caching" else 0

    build_flags = f"USE_MEMORY_ONLY={use_memory_only} USE_AES=1 USE_OCCLUM=1 USE_SYS_TIME={use_sys_time} USE_SYS_TIME_OPERATORS={use_sys_time_operators} NUM_TOKENS={num_tokens}"
    run_command_without_output(f"make clean && make {build_flags} occlum_server", cwd=f"{server_with_tls_path}/src")
    run_command_without_output(f"make clean && make {build_flags} USE_FILE_CACHING={use_file_caching}", cwd=server_with_tls_path)

def modify_occlum_json(user_space):
    file_path = f'{path_to_occlum}/occlum_workspace/Occlum.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    data['resource_limits']['user_space_size'] = user_space
    data['resource_limits']['kernel_space_heap_size'] = "128MB"
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"'user_space_size' updated to {data['resource_limits']['user_space_size']}")

def extract_hex_numbers(text):
    start_time = tme.time()
    pattern = r"Message from server: \d+ ((?:[a-fA-F0-9]+\s*)+)\n Connection was closed gracefully"
    match = re.search(pattern, text)
    print("pattern matched")
    
    if match:
        start_time2 = tme.time()
        with open(tag_file_path, 'w', buffering=65536) as f:
            f.write("\n".join(match.group(1).strip().split()))
    else:
        print("No hex numbers found.")
    
def client_side(partition_folder, unique_id):
    is_llm = "gpt" in path[unique_id] or "llama" in path[unique_id] or "mistral" in path[unique_id] or "qwen" in path[unique_id]
    if is_llm:
        tme.sleep(200)
    else:
        tme.sleep(65)

    print(f"\nCurrent directory: {os.getcwd()}")

    path_ = f"{inferONNX_path}/models/{path[unique_id]}"
    input_file = f"{path_}test_data_set_0/input_0.pb"
    if is_llm:
        input_file = f"{path_}test_data_set_0/tokenizer.json"

    command = f"{server_with_tls_path}/ssl_client models {input_file} {path_}{partition_folder}"
    result = run_command_with_output(command)

    if "on_disk" in configuration:
        print("In here")
        extract_hex_numbers(result)
        tag_file = tag_file_path
    else:
        tag_file = ""

    print("In hereeee")

    command = f"{server_with_tls_path}/ssl_client inputs 1 {tag_file} {input_file}"
    run_command_without_output(command)
    close_connection()

def extract_time(text, num_tokens):
    if num_tokens == 0:
        lines = text.splitlines()
        start_model_index = next((i for i, line in enumerate(lines) if line.startswith("Model name:")), -1)
        if start_model_index == -1:
            raise ValueError("No 'Model name:' line found.")

        lines_inference = lines[start_model_index:]
        lines_inference = [line for line in lines_inference if not line.startswith("Model name:")]
        end_model_index = next((i for i, line in enumerate(lines_inference) if line.startswith("Write to client:")), len(lines_inference))
        start_index = next((i + 1 for i, line in enumerate(lines_inference) if line.startswith("Response:")), len(lines_inference))
        closing_index = next((i for i, line in enumerate(lines_inference) if line.startswith("Closing the connection...")), len(lines_inference))

        return "\n".join(lines_inference[:end_model_index] + lines_inference[start_index:closing_index])
    
    lines = text.splitlines()
    after_indices = [i for i, line in enumerate(lines) if "After onig_end" in line]
    if not after_indices:
        raise ValueError("No 'After onig_end' line found.")

    end_idx = after_indices[-1]
    start_idx = after_indices[-2] + 1 if len(after_indices) > 1 else 0
    lines_inference = lines[start_idx:end_idx]

    filtered = [
        line for line in lines_inference if any(key in line for key in ["Partition_", "Inference time"])
    ]

    lines_after_inference = lines[end_idx + 1:]
    start_index = next((i + 1 for i, line in enumerate(lines_after_inference) if line.startswith("Response:")), len(lines_after_inference))
    closing_index = next((i for i, line in enumerate(lines_after_inference) if line.startswith("Closing the connection...")), len(lines_after_inference))

    return "\n".join(filtered + lines_after_inference[start_index:closing_index])

def manage_connection():
    unique_id = 0
    for model_name in path:
        modify_occlum_json(occlum_user_space[unique_id])

        if configuration == "memory_only_operators":
            with open(f"{inferONNX_path}/memory_intensive_ops/{model_name[:-1]}.txt", 'a') as file:
                file.write("\nSGX\n----\n")

        num_tokens = 0
        if "gpt" in model_name or "llama" in model_name or "mistral" in model_name or "qwen" in model_name:
            num_tokens = number_of_tokens

        for i in range(num_runs):
            init_client(0, num_tokens)
            if configuration == "on_disk":
                run_command("sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
            client = threading.Thread(args=(partition_folder, unique_id),target=client_side)
            client.start()

            command = f"cp {server_with_tls_path}/src/./occlum_server image/bin && occlum build && occlum run /bin/occlum_server"
            if configuration == "memory_only_operators":
                command += f" >> {inferONNX_path}/memory_intensive_ops/{model_name[:-1]}.txt"

            run_command(command, cwd=f"{path_to_occlum}/occlum_workspace")
            client.join()

            if configuration == "memory_only_operators":
                continue
            
            # init_client(1, num_tokens)
            # if configuration == "on_disk":
            #     run_command("sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
            # client = threading.Thread(args=(partition_folder, unique_id),target=client_side)
            # client.start()

            # command = f"cp {server_with_tls_path}/src/./occlum_server image/bin && occlum build && occlum run /bin/occlum_server"
            # result = run_command_with_output(command, cwd=f"{path_to_occlum}/occlum_workspace")
            # client.join()
            # inference_times = extract_time(result, num_tokens)
            
            # if configuration == "memory_only":
            #     file_path = f"{server_with_tls_path}/inference_time_in_occlum_memory_only_aes.txt"
            # elif configuration == "on_disk_caching":
            #     file_path = f"{server_with_tls_path}/inference_time_in_occlum_on_disk_aes_file_caching.txt"
            # else:
            #     file_path = f"{server_with_tls_path}/inference_time_in_occlum_on_disk_aes.txt"

            # with open(file_path, 'a') as file:
            #     file.write(inference_times + "\n")
        unique_id += 1

def close_connection():
    output = subprocess.Popen([f"{server_with_tls_path}/ssl_client quit"], stdout=subprocess.PIPE, shell=True)
    output.wait()

def main():
    if (len(sys.argv) != 6 and len(sys.argv) != 7) or sys.argv[1] not in ["memory_only", "memory_only_operators", "on_disk", "on_disk_caching"] or (sys.argv[2] != "entire" and "partitions" not in sys.argv[2]) or (sys.argv[1] == "memory_only" and sys.argv[2] == "partitions"):
        print("Usage: python3 run_models_in_occlum.py <memory_only/memory_only_operators/on_disk/on_disk_caching> <entire/partitions only for disk> <number_of_tokens> <number_of_runs> <path_to_inferONNX> <''/model_name_for_memory_only_operators>")
        exit(1)

    global configuration, entire_or_partition, number_of_tokens, num_runs, inferONNX_path
    configuration = sys.argv[1]
    entire_or_partition = sys.argv[2]
    number_of_tokens = int(sys.argv[3])
    num_runs = int(sys.argv[4])
    inferONNX_path = sys.argv[5]
    model_folder = ""
    if len(sys.argv) == 7:
        model_folder = sys.argv[6]
        assert model_folder != "", "Model folder is not provided" 

    if inferONNX_path == "./":
        inferONNX_path = os.getcwd()

    global path_to_occlum, server_with_tls_path, tag_file_path
    path_to_occlum = os.path.join(inferONNX_path, "..")
    server_with_tls_path = os.path.join(inferONNX_path, "src/server_with_tls")
    tag_file_path = os.path.join(server_with_tls_path, "tag_file.txt")

    global partition_folder, occlum_user_space
    partition_folder = entire_or_partition if "partitions" in entire_or_partition else ""
    
    # Max capacity: 23GB
    global path
    if model_folder == "":
        path = [
            "gpt2/", "smol-llama-220M-GQA/", "mistral-300M/", "qwen2.5-0.5B/"
        ]
        occlum_user_space = ["5GB", "6GB", "9GB", "16GB"]
    else:
        if "gpt" in model_folder:
            occlum_user_space = ["5GB"]
        elif"llama" in model_folder:
            occlum_user_space = ["6GB"]
        elif "" in model_folder:
            occlum_user_space = ["9GB"]
        else:
            occlum_user_space = ["16GB"]


    manage_connection()
    run_command_without_output("make clean", cwd=server_with_tls_path)
    run_command_without_output("make clean", cwd=f"{server_with_tls_path}/src")


if __name__ == "__main__":
    main()