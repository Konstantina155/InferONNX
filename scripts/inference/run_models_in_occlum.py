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
    output = subprocess.Popen(cmd, cwd=cwd, stderr=subprocess.PIPE, shell=True)
    _, result_stderr = output.communicate()
    return result_stderr.decode('utf-8')

def run_command_without_output(cmd, cwd=None):
    process = subprocess.Popen(cmd, cwd=cwd, shell=True)
    process.wait()

def init_client(use_sys_time):
    use_sys_time_operators = 1 if configuration == "memory_only_operators" else 0
    use_memory_only = 0 if configuration == "on_disk" else 1

    build_flags = f"USE_MEMORY_ONLY={use_memory_only} USE_AES=1 USE_OCCLUM=1 USE_SYS_TIME={use_sys_time} USE_SYS_TIME_OPERATORS={use_sys_time_operators}"
    run_command_without_output(f"make clean && make {build_flags} occlum_server", cwd=f"{server_with_tls_path}/src")
    run_command_without_output(f"make clean && make {build_flags}", cwd=server_with_tls_path)

def modify_occlum_json(user_space):
    file_path = f'{path_to_occlum}/occlum_workspace/Occlum.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    data['resource_limits']['user_space_size'] = user_space
    data['resource_limits']['kernel_space_heap_size'] = "64MB"
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"'user_space_size' updated to {data['resource_limits']['user_space_size']}")

def extract_hex_numbers(text):
    pattern = r"Message from server: \d+ ((?:[a-fA-F0-9]+\s*)+)\n Connection was closed gracefully"
    match = re.search(pattern, text)
    
    if match:
        hex_numbers = match.group(1).strip().split()

        with open(tag_file_path, 'w') as f:
            f.write("\n".join(hex_numbers))
    else:
        print("No hex numbers found.")
    
def client_side(partition_folder, unique_id):
    tme.sleep(65)

    print(f"\nCurrent directory: {os.getcwd()}")

    path_ = f"{inferONNX_path}/models/{path[unique_id]}"
    

    command = f"{server_with_tls_path}/ssl_client models {path_}test_data_set_0/input_0.pb {path_}{partition_folder}"
    result = run_command_with_output(command)

    if configuration == "on_disk":
        extract_hex_numbers(result)
        tag_file = tag_file_path
    else:
        tag_file = ""

    command = f"{server_with_tls_path}/ssl_client inputs 1 {tag_file} {path_}test_data_set_0/input_0.pb"
    run_command_without_output(command)
    close_connection()

def extract_time(text):
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

def manage_connection():
    unique_id = 0
    for model_name in path:
        modify_occlum_json(occlum_user_space[unique_id])

        if configuration == "memory_only_operators":
            with open(f"{inferONNX_path}/memory_intensive_ops/{model_name[:-1]}.txt", 'a') as file:
                file.write("\nSGX\n----\n")

        for i in range(num_runs):
            init_client(0)
            client = threading.Thread(args=(partition_folder, unique_id),target=client_side)
            client.start()

            command = f"cp {server_with_tls_path}/src/./occlum_server image/bin && occlum build && occlum run /bin/occlum_server"
            if configuration == "memory_only_operators":
                command += f" >> {inferONNX_path}/memory_intensive_ops/{model_name[:-1]}.txt"

            run_command(command, cwd=f"{path_to_occlum}/occlum_workspace")
            client.join()

            if configuration == "memory_only_operators":
                continue

            init_client(1)
            client = threading.Thread(args=(partition_folder, unique_id),target=client_side)
            client.start()

            command = f"cp {server_with_tls_path}/src/./occlum_server image/bin && occlum build && occlum run /bin/occlum_server"
            result = run_command_with_output(command, cwd=f"{path_to_occlum}/occlum_workspace")
            client.join()
            inference_times = extract_time(result)
            
            if configuration == "memory_only":
                file_path = f"{server_with_tls_path}/inference_time_in_occlum_memory_only_aes.txt"
            else:
                file_path = f"{server_with_tls_path}/inference_time_in_occlum_on_disk_aes.txt"

            with open(file_path, 'a') as file:
                file.write(inference_times + "\n")            
        unique_id += 1

def close_connection():
    output = subprocess.Popen([f"{server_with_tls_path}/ssl_client quit"], stdout=subprocess.PIPE, shell=True)
    output.wait()

def main():
    if len(sys.argv) != 5 or sys.argv[1] not in ["memory_only", "memory_only_operators", "on_disk"] or (sys.argv[2] != "entire" and "partitions" not in sys.argv[2]) or (sys.argv[1] == "memory_only" and sys.argv[2] == "partitions"):
        print("Usage: python3 run_models_in_occlum.py <memory_only/on_disk> <entire/partitions only for disk> <number_of_runs> <path_to_inferONNX>")
        exit(1)

    global configuration, entire_or_partition, num_runs, inferONNX_path
    configuration = sys.argv[1]
    entire_or_partition = sys.argv[2]
    num_runs = int(sys.argv[3])
    inferONNX_path = sys.argv[4]

    if inferONNX_path == "./":
        inferONNX_path = os.getcwd()

    global path_to_occlum, server_with_tls_path, tag_file_path
    path_to_occlum = os.path.join(inferONNX_path, "..")
    server_with_tls_path = os.path.join(inferONNX_path, "src/server_with_tls")
    tag_file_path = os.path.join(server_with_tls_path, "tag_file.txt")
    global path
    path = [
        "squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", 
        "efficientnet-lite4-11/", "inception-v3-12/", 
        "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2-l-18/"
    ]

    global partition_folder, occlum_user_space
    partition_folder = entire_or_partition if "partitions" in entire_or_partition else ""
    occlum_user_space = ["300MB", "300MB", "300MB", "400MB", "700MB", "2GB", "2GB", "3GB"]


    manage_connection()
    run_command_without_output("make clean", cwd=server_with_tls_path)
    run_command_without_output("make clean", cwd=f"{server_with_tls_path}/src")


if __name__ == "__main__":
    main()