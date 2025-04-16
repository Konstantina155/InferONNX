import subprocess
import os
import sys
import threading
import time as tme
import re

if len(sys.argv) != 4 or sys.argv[1] not in ["memory_only", "memory_only_operators", "on_disk", "tls_memory_only"]:
    print("Usage: python3 run_models_in_cpu.py <memory_only/memory_only_operators/on_disk/tls_memory_only> <number_of_runs> <path_to_inferONNX>")
    exit(1)

try:
    num_runs = int(sys.argv[2])
except ValueError:
    print("Usage: python3 run_models_in_cpu.py <memory_only/memory_only_operators/on_disk/tls_memory_only> <number_of_runs> <path_to_inferONNX>")
    exit(1)

configuration = sys.argv[1]
inferONNX_path = sys.argv[3]
path_to_occlum = inferONNX_path + "/.."
server_with_tls_path = inferONNX_path + "/src/server_with_tls"
server_without_tls_path = inferONNX_path + "/src/server_without_tls"
tag_file_path = server_without_tls_path + "/tag_file.txt"
path = ["squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", "efficientnet-lite4-11/", "inception-v3-12/", "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2-l-18/"]

previous_path = os.getcwd()

def init_server_client():
    os.chdir(f"{server_with_tls_path}/src")
    command = f"make clean && make USE_AES=0 USE_OCCLUM=0 USE_SYS_TIME=1 server"
    print(f"Command: {command}")
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (out, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

    os.chdir(server_with_tls_path)
    command = f"make clean && make USE_AES=0 USE_OCCLUM=0 USE_SYS_TIME=0"
    print(f"Command: {command}")
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (out, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

def init(use_aes):
    use_sys_time_operators=0
    if configuration == "on_disk":
        use_memory_only = 0
    elif configuration == "memory_only_operators":
        use_sys_time_operators=1
        use_memory_only = 1
    else:
        use_memory_only = 1
    os.chdir(server_without_tls_path)
    command = f"make clean && make USE_AES={use_aes} USE_MEMORY_ONLY={use_memory_only} USE_SYS_TIME_OPERATORS={use_sys_time_operators}"
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (out, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

    os.chdir(f"{server_without_tls_path}/src")
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (out, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)
        
    os.chdir(f"{inferONNX_path}/../")

def extract_hex_numbers(text):
    text = text.decode('utf-8')

    pattern = r"Message from server: \d+ ((?:[a-fA-F0-9]+\s*)+)\n"
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

def client_side(unique_id):
    if configuration == "tls_memory_only":
        client_command = f"{server_with_tls_path}/./ssl_client"
    else:
        client_command = f"{server_without_tls_path}/./client"
    
    tme.sleep(2)

    current_dir = os.getcwd()
    print(f"\nCurrent directory: {current_dir}")

    path_ = f"{inferONNX_path}/models/" + path[unique_id]

    if configuration == "on_disk":
        output = subprocess.Popen("sudo sysctl -w vm.drop_caches=3", stdout=subprocess.PIPE, shell=True)
        (out, err) = output.communicate()
        if err:
            print(f"Error: {err.decode()}")
            exit(1)

    command = client_command + " models " +  path_ + "test_data_set_0/input_0.pb " + path_
    print(command)
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    time, err = output.communicate()

    if configuration == "on_disk":
        output = subprocess.Popen("sudo sysctl -w vm.drop_caches=3", stdout=subprocess.PIPE, shell=True)
        (out, err) = output.communicate()
        if err:
            print(f"Error: {err.decode()}")
            exit(1)
        tag_file = ""
    elif "aes" in configuration:
        extract_hex_numbers(time)
        tag_file = tag_file_path
    else:
        tag_file = ""

    command = client_command + " inputs 1 " + tag_file + " " + path_ + "test_data_set_0/input_0.pb"
    print(command)
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (pred, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

    close_connection()

def manage_connection():
    unique_id = 0
    for model_name in path:

        if configuration == "memory_only_operators":
            with open(f"{inferONNX_path}/memory_intensive_ops/{model_name[:-1]}.txt", 'a') as file:
                file.write("\nCPU\n----\n\n")

        for i in range(num_runs):
            client = threading.Thread(args=(unique_id,),target=client_side)
            client.start()

            command = ""
            if configuration == "tls_memory_only":
                os.chdir(f"{path_to_occlum}/occlum_workspace/")
                command = f"{server_with_tls_path}/src/./server"
            elif configuration == "memory_only_operators":
                os.chdir(f"{server_without_tls_path}/src")
                command = f"./server >> {inferONNX_path}/memory_intensive_ops/{model_name[:-1]}.txt"
            else:
                os.chdir(f"{server_without_tls_path}/src")
                command = "./server"
            subprocess.run(command, shell=True)
            client.join()

        unique_id += 1

def close_connection():
    if configuration == "tls_memory_only":
        command = f"{server_with_tls_path}/./ssl_client quit"
    else:
        command = f"{server_without_tls_path}/./client quit"
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    output.wait()

if __name__ == "__main__":
    if configuration != "tls_memory_only":
        init(0)
    else:
        init_server_client()

    manage_connection()

    if configuration != "tls_memory_only":
        os.chdir(server_without_tls_path)
        os.system("make clean")
        os.chdir(f"{server_without_tls_path}/src")
        os.system("make clean")
    else:
        os.chdir(server_with_tls_path)
        os.system("make clean")
        os.chdir(f"{server_with_tls_path}/src")
        os.system("make clean")

    os.chdir(previous_path)