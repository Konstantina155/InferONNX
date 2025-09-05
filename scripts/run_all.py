import os
import sys
import subprocess

def check_and_create_dir(directory):
    os.makedirs(directory, exist_ok=True)

def init_occlum_cert():
    os.system("occlum init")
    os.mkdir("image/bin/encrypted_models")
    os.system(f"openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout ../certificates/key.pem \
        -out ../certificates/cert.pem \
        -days 365 \
        -subj \"/C=US/ST=CA/L=SanFrancisco/O=MyCompany/CN=localhost\"")

    os.system(f"cp ../certificates/* image/bin")

def create_tag_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

def run_command(command):
    subprocess.run(command, shell=True, check=True)
    
def run_all_and_create_plot(path_to_scripts, number_of_runs, partitions_folder, inferONNX_path):
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum.py on_disk_caching entire {number_of_runs} {inferONNX_path}")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum.py on_disk entire {number_of_runs} {inferONNX_path}")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum.py on_disk {partitions_folder} {number_of_runs} {inferONNX_path}")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum.py memory_only entire {number_of_runs} {inferONNX_path}")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_cpu.py tls_memory_only {number_of_runs} {inferONNX_path}")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_cpu.py tls_on_disk {number_of_runs} {inferONNX_path}")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_cpu.py on_disk {number_of_runs} {inferONNX_path}")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_cpu.py memory_only {number_of_runs} {inferONNX_path}")

    run_command(f"python3 {path_to_scripts}/create_plots.py {inferONNX_path} {partitions_folder} {number_of_runs}")

def measure_each_op_time(path_to_scripts, inferONNX_path):
    os.makedirs(f"{inferONNX_path}/memory_intensive_ops", exist_ok=True)
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_cpu.py memory_only_operators 1 {inferONNX_path}")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum.py memory_only_operators entire 1 {inferONNX_path}")

def clean_up(tag_files, inferONNX_path, home_dir):
    for tag_file in tag_files:
        os.remove(tag_file)

    os.chdir(inferONNX_path)
    run_command(f'rm -rf ../occlum_workspace ../certificates {home_dir}/unencrypted_models {home_dir}/encrypted_models')
    run_command(f'rm src/server_with_tls/inference_time_* src/server_without_tls/inference_time_*')

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 run_all.py <partitions_folder> <number_of_runs>")
        exit(1)

    partitions_folder = sys.argv[1]
    number_of_runs = sys.argv[2]

    home_dir = os.getenv("HOME")
    os.chdir("..")
    check_and_create_dir('occlum_workspace')
    check_and_create_dir('certificates')
    check_and_create_dir(f'{home_dir}/unencrypted_models')
    check_and_create_dir(f'{home_dir}/encrypted_models')
    os.chdir("occlum_workspace")

    if not os.listdir('./'):
        init_occlum_cert()

    inferONNX_path = os.getcwd() + "/../InferONNX"
    path_to_scripts = inferONNX_path + "/scripts"
    tag_tls_server = inferONNX_path + '/src/server_with_tls/tag_file.txt'
    tag_no_tls_server = inferONNX_path + '/src/server_without_tls/tag_file.txt'
    create_tag_file(tag_tls_server)
    create_tag_file(tag_no_tls_server)

    run_all_and_create_plot(path_to_scripts, number_of_runs, partitions_folder, inferONNX_path)
    measure_each_op_time(path_to_scripts, inferONNX_path)

    clean_up([tag_no_tls_server, tag_tls_server], inferONNX_path, home_dir)

if __name__ == "__main__":
    main()