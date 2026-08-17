import os
import sys
import subprocess

def run_inference(directory, test_path):
    command = f"env ORT_DYLIB_PATH={home_directory}/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so ./standalone_inference {directory} {test_path} ../../../models/distilbert-base-finetuned/model.onnx ../../../models/distilbert-base-finetuned/tokenizer.json prompt.txt"
    try:
        output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, out_stderr = output.communicate()
        output = out_stderr.decode("utf-8")
        
        begin_of_max = output.rfind("Inference:")
        if begin_of_max != -1:
            end_of_sentence = output.find("Next_token_id:", begin_of_max)
            if end_of_sentence != -1:
                inference_result = output[begin_of_max + 11 : end_of_sentence].rstrip()
                inference_result += " ..."
                return inference_result
    except subprocess.CalledProcessError as error:
        print(f"Inference failed: {error}")
        raise

def run_command(command):
    try:
        output = subprocess.Popen(command, shell=True)
        output.wait() 
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise

def make_build(num_tokens):
    run_command("make clean")
    run_command(f"make NUM_TOKENS={num_tokens}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 check_accuracy.py <inter_partitions_folder> <intra_partitions_folder> <num_tokens>")
        exit(1)

    global home_directory
    home_directory = os.path.expanduser("~")
    path = ["gpt2", "smol-llama-220M-GQA", "mistral-300M", "qwen2.5-0.5B"]
    partitions_inter = sys.argv[1]
    partitions_intra = sys.argv[2]
    num_tokens = sys.argv[3]
    previous_path = os.getcwd()
    os.chdir("src/server_with_tls/scripts")
    make_build(num_tokens)

    for model_name in path:
        test_path = f"../../../models/{model_name}/test_data_set_0/tokenizer.json"
        inference_inter_operators = run_inference(f"../../../models/{model_name}/{partitions_inter}", test_path)
        inference_intra_operators = run_inference(f"../../../models/{model_name}/{partitions_intra}", test_path)
        inference_whole = run_inference(f"../../../models/{model_name}/", test_path)

        print(f"Inference result for {model_name} for {num_tokens} tokens:")
        print("Full model execution:", inference_whole)
        print("Inter-operator partitioning:", inference_inter_operators)
        print("Intra-operator partitioning:", inference_intra_operators)
        print()

    run_command("make clean")
    os.chdir(previous_path)

if __name__ == "__main__":
    main()