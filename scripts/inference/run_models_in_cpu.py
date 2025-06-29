import subprocess
import os
import sys
import threading
import time as tme
import re

def run_command(cmd, cwd=None):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("STDOUT:\n", result.stdout.decode())
        print("STDERR:\n", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print("Command failed with error:")
        print(e.stderr.decode())

def run_command_without_output(cmd, cwd=None):
    process = subprocess.Popen(cmd, cwd=cwd, shell=True)
    process.wait()

def init_server_client(num_tokens):
    run_command_without_output(f"make clean && make USE_AES=0 USE_OCCLUM=0 USE_SYS_TIME=1 NUM_TOKENS={num_tokens} server", cwd=f"{server_with_tls_path}/src")
    run_command_without_output(f"make clean && make USE_AES=0 USE_OCCLUM=0 USE_SYS_TIME=0 NUM_TOKENS={num_tokens}", cwd=server_with_tls_path)

def init(use_aes):
    use_sys_time_operators=0
    use_memory_only = 0
    if configuration == "memory_only_operators":
        use_sys_time_operators=1
        use_memory_only = 1
    elif configuration != "on_disk":
        use_memory_only = 1

    command = f"make clean && make USE_AES={use_aes} USE_MEMORY_ONLY={use_memory_only} USE_SYS_TIME_OPERATORS={use_sys_time_operators}"
    run_command_without_output(command, cwd=server_without_tls_path)
    run_command_without_output(command, cwd=f"{server_without_tls_path}/src")

def client_side(unique_id):
    client_command = f"{server_with_tls_path}/./ssl_client" if configuration == "tls_memory_only" else f"{server_without_tls_path}/./client"
    
    tme.sleep(2)

    print(f"\nCurrent directory: {os.getcwd()}")

    path_ = f"{inferONNX_path}/models/{path[unique_id]}"
    input_file = f"{path_}test_data_set_0/input_0.pb"
    is_llm = "albert" in path[unique_id] or "gpt2" in path[unique_id]
    if is_llm:
        input_file = f"{path_}test_data_set_0/tokenizer.json"

    if configuration == "on_disk":
        run_command("sudo sysctl -w vm.drop_caches=3")

    command = f"{client_command} models {input_file} {path_}"
    run_command_without_output(command)

    if configuration == "on_disk":
        run_command("sudo sysctl -w vm.drop_caches=3")

    command = f"{client_command} inputs 1 {input_file}"
    run_command_without_output(command)
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

            cwd = f"{server_with_tls_path}/src/" if configuration == "tls_memory_only" else f"{server_without_tls_path}/src/"
            command = "./server"
            if configuration == "memory_only_operators":
                command += f" >> ../../../memory_intensive_ops/{model_name[:-1]}.txt"
            run_command(command, cwd=cwd)
            
            client.join()

        unique_id += 1

def close_connection():
    client_command = f"{server_with_tls_path}/./ssl_client quit" if configuration == "tls_memory_only" else f"{server_without_tls_path}/./client quit"
    run_command_without_output(client_command)

def main():
    if len(sys.argv) != 5 or sys.argv[1] not in ["memory_only", "memory_only_operators", "on_disk", "tls_memory_only"]:
        print("Usage: python3 run_models_in_cpu.py <memory_only/memory_only_operators/on_disk/tls_memory_only> <number_of_runs> <path_to_inferONNX> <num_tokens>")
        exit(1)

    global configuration, num_runs, inferONNX_path, num_tokens
    configuration = sys.argv[1]
    num_runs = int(sys.argv[2])
    inferONNX_path = sys.argv[3]
    num_tokens = int(sys.argv[4])

    global server_with_tls_path, server_without_tls_path
    server_with_tls_path = os.path.join(inferONNX_path, "src/server_with_tls")
    server_without_tls_path = os.path.join(inferONNX_path, "src/server_without_tls")
    global path
    path = [
        "squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", 
        "efficientnet-lite4-11/", "inception-v3-12/", 
        "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2-l-18/",
    ]

    if configuration != "tls_memory_only":
        init(0)
    else:
        init_server_client(num_tokens)

    manage_connection()

    if configuration != "tls_memory_only":
        run_command_without_output("make clean", cwd=server_without_tls_path)
        run_command_without_output("make clean", cwd=f"{server_without_tls_path}/src")
    else:
        run_command_without_output("make clean", cwd=server_with_tls_path)
        run_command_without_output("make clean", cwd=f"{server_with_tls_path}/src")

    
if __name__ == "__main__":
    main()