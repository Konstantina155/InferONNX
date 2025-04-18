import subprocess
import os
import threading
import json
import time as tme
import sys
import re
from natsort import natsorted

if len(sys.argv) != 5 or sys.argv[1] not in ["memory_only", "memory_only_operators", "on_disk"] or sys.argv[2] not in ["entire", "partitions"] or (sys.argv[1] == "memory_only" and sys.argv[2] == "partitions"):
    print("Usage: python3 run_models_in_occlum.py <memory_only/on_disk> <entire/partitions only for disk> <number_of_runs> <path_to_inferONNX>")
    exit(1)

try:
    num_runs = int(sys.argv[3])
except ValueError:
    print("Usage: python3 run_models_in_occlum.py <memory_only/on_disk> <entire/partitions only for disk> <number_of_runs> <path_to_inferONNX>")
    exit(1)

configuration = sys.argv[1]
entire_or_partition = sys.argv[2]
inferONNX_path = sys.argv[4]
path_to_occlum = inferONNX_path + "/.."
server_with_tls_path = inferONNX_path + "/src/server_with_tls"
tag_file_path = server_with_tls_path + "/tag_file.txt"
path = ["squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", "efficientnet-lite4-11/", "inception-v3-12/", "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2-l-18/"]
if entire_or_partition == "partitions":
    partition_folder = "partitions/"
else:
    partition_folder = ""
occlum_user_space = ["300MB", "300MB", "300MB", "400MB", "700MB", "2GB", "2GB", "3GB"]

previous_path = os.getcwd()
print(previous_path)

def form_array_of_partitions_operators(path):
    file_list = os.listdir(path)
    onnx_files = [f for f in file_list if f.endswith('.onnx')]
    return natsorted(onnx_files)

def init_client(use_sys_time):
    use_sys_time_operators = 0
    if configuration == "memory_only":
        use_memory_only = 1
    elif configuration == "memory_only_operators":
        use_memory_only = 1
        use_sys_time_operators=1
    else:
        use_memory_only = 0
    
    command = f"make clean && make USE_MEMORY_ONLY={use_memory_only} USE_AES=1 USE_OCCLUM=1 USE_SYS_TIME={use_sys_time} USE_SYS_TIME_OPERATORS={use_sys_time_operators}"
    os.chdir(server_with_tls_path)
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (pred, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

    os.chdir(f"{server_with_tls_path}/src")
    command = f"make clean && make USE_MEMORY_ONLY={use_memory_only} USE_AES=1 USE_OCCLUM=1 USE_SYS_TIME={use_sys_time} USE_SYS_TIME_OPERATORS={use_sys_time_operators} occlum_server"
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (pred, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

def modify_occlum_json(user_space):
    file_path = f'{path_to_occlum}/occlum_workspace/Occlum.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    data['resource_limits']['user_space_size'] = user_space
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"'user_space_size' updated to {data['resource_limits']['user_space_size']}")

def extract_hex_numbers(text):
    text = text.decode('utf-8')
    print("Text: ", text)
    pattern = r"Message from server: \d+ ((?:[a-fA-F0-9]+\s*)+)\n Connection was closed gracefully"
    match = re.search(pattern, text)
    
    if match:
        hex_numbers = match.group(1).strip().split()

        f = open(tag_file_path, 'w')
        for hex_num in hex_numbers:
            hex_number = hex_num.strip()
            f.write(hex_number)
            f.write('\n')
        f.close()
    else:
        print("No hex numbers found.")
    
def client_side(partition_folder, unique_id):
    os.chdir(f"{path_to_occlum}/occlum_workspace")
    current_dir = os.getcwd()
    print(f"\nCurrent directory: {current_dir}")

    path_ = inferONNX_path + "/models/" + path[unique_id]
    tme.sleep(65)

    command = f"{server_with_tls_path}/ssl_client models " + path_ + "test_data_set_0/input_0.pb " + path_ + partition_folder
    print(command)
    output = subprocess.Popen([command], stderr=subprocess.PIPE, shell=True)
    _, time = output.communicate()
    if not time:
        print(f"Error: {time.decode()}")
    
    if "disk" in configuration:
        extract_hex_numbers(time)
        tag_file = tag_file_path
    else:
        tag_file = ""

    command = f"{server_with_tls_path}/ssl_client inputs 1 " + tag_file + " " + path_ + "test_data_set_0/input_0.pb"
    print(command)
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (pred, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)
    close_connection()

def extract_time(text):
    text = text.stderr.decode('utf-8')
    print("Text before extraction:")
    print(text)
    print("End of text before extraction.")

    start_model_index = end_model_index = start_index = closing_line_index = -1
    lines = text.splitlines()
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("Model name:"):
            start_model_index = i
            break

    if start_model_index is None:
        raise ValueError("No 'Model name:' line found that matches the conditions.")

    lines_inference = lines[start_model_index:]
    start_model_index = -1
    for i, line in enumerate(lines_inference):
        if line.startswith("Model name:"):
            if start_model_index == -1:
                start_model_index = i
            lines_inference.remove(line)
        if line.startswith("Write to client:"):
            end_model_index = i
        elif line.startswith("Response:"):
            start_index = i + 1
        elif line.startswith("Closing the connection..."):
            closing_line_index = i

    extracted_lines = lines_inference[start_model_index:end_model_index] + lines_inference[start_index:closing_line_index]
    return "\n".join(extracted_lines)

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
                command += f">> {inferONNX_path}/memory_intensive_ops/{model_name[:-1]}.txt"

            load_models = subprocess.run(command, shell=True)
            client.join()

            if configuration == "memory_only_operators":
                continue

            init_client(1)
            client = threading.Thread(args=(partition_folder, unique_id),target=client_side)
            client.start()

            inference = subprocess.run(f"cp {server_with_tls_path}/src/./occlum_server image/bin && occlum build && occlum run /bin/occlum_server", stderr=subprocess.PIPE, shell=True)
            client.join()
            inference_times = extract_time(inference)
            
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

if __name__ == "__main__":
    manage_connection()
    os.chdir(server_with_tls_path)
    os.system("make clean")
    os.chdir(f"{server_with_tls_path}/src")
    os.system("make clean")
    os.chdir(previous_path)