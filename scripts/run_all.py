import os
import sys
import json
import random
import numpy as np
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
        -subj \"/C=US/ST=CA/L=SanFrancisco/O=MyCompany/CN=localhost\"") #server's IP when client in different machine

    os.system("cp ../certificates/* image/bin")

def create_tag_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def select_evenly_by_length(prompts, number_of_prompts):
    prompts = sorted(prompts, key=len) 
    if len(prompts) <= number_of_prompts:
        return prompts
    indices = np.linspace(0, len(prompts) - 1, number_of_prompts, dtype=int) 
    return [prompts[i] for i in indices]

def create_selected_prompts_json(inferONNX_path, number_of_prompts):
    prompts = []
    with open(f"{inferONNX_path}/models/1english_openpii_8k.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt = row["source_text"]
            if prompt.strip(): prompts.append(prompt)

    lengths = np.array([len(p) for p in prompts])

    # Dataset info
    print("Number of prompts:", len(prompts))
    print("Min:", lengths.min())
    print("Max:", lengths.max())
    print("Mean:", lengths.mean())
    print("Median:", np.median(lengths))

    # Divide into three groups
    p33 = np.percentile(lengths, 33)
    p67 = np.percentile(lengths, 67)
    short = [p for p in prompts if len(p) <= p33]
    medium = [p for p in prompts if p33 < len(p) <= p67]
    long = [p for p in prompts if len(p) > p67]
    print("\nGroup boundaries:")
    print(f"Short: <= {p33:.0f} characters, {len(short)} total prompts")
    print(f"Medium: {(p33+1):.0f} - {p67:.0f} characters, {len(medium)} total prompts")
    print(f"Long: > {p67:.0f} characters, {len(long)} total prompts")

    # Select prompts evenly across each group's length range
    base = number_of_prompts // 3
    remainder = number_of_prompts % 3
    short_num = base + (1 if remainder >= 1 else 0)
    medium_num = base + (1 if remainder >= 2 else 0)
    long_num = base
    long_num = number_of_prompts - short_num - medium_num
    print(short_num, medium_num, long_num)
    selected_short = select_evenly_by_length(short, short_num)
    selected_medium = select_evenly_by_length(medium, medium_num)
    selected_long = select_evenly_by_length(long, long_num)

    final_prompts = ( selected_short + selected_medium + selected_long )
    random.seed(42)
    random.shuffle(final_prompts)
    for p in final_prompts:
        print(len(p))

    with open(f"{inferONNX_path}/models/1english_selected_prompts.jsonl", "w", encoding="utf-8") as f:
        for p in final_prompts:
            f.write(json.dumps({
                "prompt": p,
                "length_chars": len(p)
            }, ensure_ascii=False) + "\n")
    
def run_all_and_create_plot(path_to_scripts, number_of_runs, inter_partitions_folder, intra_partitions_folder, inferONNX_path):   
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum_guardrails.py memory_only entire 1 {number_of_runs} {inferONNX_path} no_guardrails")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum_guardrails.py memory_only entire 1 {number_of_runs} {inferONNX_path} guardrails_regex")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum_guardrails.py memory_only entire 1 {number_of_runs} {inferONNX_path} guardrails_ner distilbert")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum_guardrails.py memory_only entire 1 {number_of_runs} {inferONNX_path} guardrails_ner bert-base")
    run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum_guardrails.py memory_only entire 1 {number_of_runs} {inferONNX_path} guardrails_ner bert-large")

    run_command(f"python3 {path_to_scripts}/create_plots.py {inferONNX_path} {inter_partitions_folder} {intra_partitions_folder} {number_of_runs}")

def measure_each_op_time(path_to_scripts, model_folder, intra_partitions_folder, inferONNX_path):
    os.makedirs(f"{inferONNX_path}/memory_intensive_ops", exist_ok=True)
    if model_folder == "":
        run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum.py memory_only_operators entire 1 1 {inferONNX_path}")
    else:
        run_command(f"python3 {path_to_scripts}/inference/run_models_in_occlum.py memory_only_operators {intra_partitions_folder} 1 1 {inferONNX_path} {model_folder}")

def clean_up(tag_files, inferONNX_path):
    for tag_file in tag_files:
        os.remove(tag_file)

    os.chdir(inferONNX_path)
    run_command('rm -rf ../occlum_workspace ../certificates ../unencrypted_models ../encrypted_models')
    #run_command(f'rm src/server_with_tls/inference_time_* src/server_without_tls/inference_time_*')

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 run_all.py both <inter_partitions_folder> <intra_partitions_folder> <number_of_runs> or \
             \n                          split_op <model_name> <intra_partitions_folder> <number_of_runs>")
        exit(1)

    option = sys.argv[1]
    assert option in ["both", "split_op"], "Option is not correct"
    model_folder = ""
    inter_partitions_folder = ""
    if option == "both":
        inter_partitions_folder = sys.argv[2]
    else:
        model_folder = sys.argv[2]
    intra_partitions_folder = sys.argv[3]
    number_of_runs = sys.argv[4]
    try:
        number_of_runs = int(number_of_runs)
    except ValueError as e:
        print(f"Arg: {number_of_runs} is not an Integer, error : {e}")
        sys.exit(1)
    assert number_of_runs > 0, "Number of runs is not a positive number"

    os.chdir("..")
    check_and_create_dir('occlum_workspace')
    check_and_create_dir('certificates')
    check_and_create_dir('unencrypted_models')
    check_and_create_dir('encrypted_models')
    os.chdir("occlum_workspace")

    if not os.listdir('./'):
        init_occlum_cert()

    inferONNX_path = os.getcwd() + "/../InferONNX"
    path_to_scripts = inferONNX_path + "/scripts"
    tag_tls_server = inferONNX_path + '/src/server_with_tls/tag_file.txt'
    tag_no_tls_server = inferONNX_path + '/src/server_without_tls/tag_file.txt'
    create_tag_file(tag_tls_server)
    create_tag_file(tag_no_tls_server)

    create_selected_prompts_json(inferONNX_path, number_of_runs) 

    #measure_each_op_time(path_to_scripts, model_folder, intra_partitions_folder, inferONNX_path)
    if option != "split_op":
        run_all_and_create_plot(path_to_scripts, number_of_runs, inter_partitions_folder, intra_partitions_folder, inferONNX_path)

    clean_up([tag_no_tls_server, tag_tls_server], inferONNX_path)

if __name__ == "__main__":
    main()