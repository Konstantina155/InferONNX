import subprocess
import os
import sys
import threading
import time as tme
import re

if len(sys.argv) != 5 or sys.argv[1] not in ["memory_only", "on_disk", "tls_memory_only"]:
    print("Usage: python3 run_tests_cpu.py <memory_only/on_disk/tls_memory_only> <number_of_runs> <path_to_inferONNX> <path_to_occlum>")
    exit(1)

try:
    num_runs = int(sys.argv[2])
except ValueError:
    print("Usage: python3 run_tests_cpu.py <memory_only/on_disk/tls_memory_only> <number_of_runs> <path_to_inferONNX> <path_to_occlum>")
    exit(1)

configuration = sys.argv[1]
inferONNX_path = sys.argv[3]
tls_server_path = inferONNX_path + "/src/tls_server"
no_tls_server_path = inferONNX_path + "/src/no_tls_server"
tag_file_path = inferONNX_path + "/src/no_tls_server/tag_file.txt"
path_to_occlum = sys.argv[4]
path = ["squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", "efficientnet-lite4-11/", "inception-v3-12/", "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2-l-18/"]
path_models = [""] * len(path)

def init_server_client():
    os.chdir(f"{tls_server_path}/src")
    command = f"make clean && make USE_AES=0 USE_OCCLUM=0 USE_SYS_TIME=1 server"
    print(f"Command: {command}")
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (out, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

    os.chdir(tls_server_path)
    command = f"make clean && make USE_AES=0 USE_OCCLUM=0 USE_SYS_TIME=0"
    print(f"Command: {command}")
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (out, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

def init(use_aes):
    if configuration == "on_disk":
        use_memory_only = 0
    else:
        use_memory_only = 1
    os.chdir(no_tls_server_path)
    command = f"make clean && make USE_AES={use_aes} USE_MEMORY_ONLY={use_memory_only}"
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (out, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

    os.chdir(f"{no_tls_server_path}/src")
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (out, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

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

def client_side(path_model, unique_id):
    if configuration == "tls_memory_only":
        client_command = f"{tls_server_path}/./ssl_client"
    else:
        client_command = f"{no_tls_server_path}/./client"
    
    tme.sleep(2)

    current_dir = os.getcwd()
    print(f"\nCurrent directory: {current_dir}")

    path_ = f"{tls_server_path}/models/" + path[unique_id]

    if configuration == "on_disk":
        output = subprocess.Popen("sudo sysctl -w vm.drop_caches=3", stdout=subprocess.PIPE, shell=True)
        (out, err) = output.communicate()
        if err:
            print(f"Error: {err.decode()}")
            exit(1)

    input_file = path_ + "test_data_set_0/input_0.pb"

    command = client_command + " models " + input_file + " " + path_ + " " + path_ + path_model
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    time, err = output.communicate()

    if configuration == "on_disk":
        output = subprocess.Popen("sudo sysctl -w vm.drop_caches=3", stdout=subprocess.PIPE, shell=True)
        (out, err) = output.communicate()
        if err:
            print(f"Error: {err.decode()}")
            exit(1)
        tag_file = ""
    elif configuration == "tls_memory_only":
        extract_hex_numbers(time)
        tag_file = tag_file_path
    else:
        tag_file = ""

    command = client_command + " inputs 1 " + tag_file + " " + input_file
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    (pred, err) = output.communicate()
    if err:
        print(f"Error: {err.decode()}")
        exit(1)

    close_connection()

def thread_to_run_client(path_model, unique_id):
    client = threading.Thread(args=(path_model,unique_id),target=client_side)
    client.start()

    if configuration == "tls_memory_only":
        os.chdir(f"{tls_server_path}/src")
    else:
        os.chdir(f"{no_tls_server_path}/src")
    subprocess.run("./server")
    client.join()

def manage_connection():
    unique_id = 0
    for path_model in path_models:
        for i in range(num_runs):
            thread_to_run_client(path_model, unique_id)
        unique_id += 1

def close_connection():
    if configuration == "tls_memory_only":
        command = f"{tls_server_path}/./ssl_client quit"
    else:
        command = f"{no_tls_server_path}/./client quit"
    output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    output.wait()

if __name__ == "__main__":
    if configuration != "tls_memory_only":
        init(0)
    else:
        init_server_client()
manage_connection()