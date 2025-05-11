import os
import sys
import subprocess

def run_inference(directory, test_path):
    command = f"./standalone_inference {directory} {test_path}"
    try:
        output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, out_stderr = output.communicate()
        output = out_stderr.decode("utf-8")
        
        begin_of_max = output.rfind("Max is")
        if begin_of_max != -1:
            end_of_sentence = output.find("!", begin_of_max)
            if end_of_sentence != -1:
                inference_result = output[begin_of_max:end_of_sentence + 1]
                print(inference_result)
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

def make_build():
    run_command("make clean")
    run_command("make")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 check_accuracy.py <partitions_folder>")
        exit(1)

    path = ["squeezenet1.0-7", "mobilenetv2-7", "densenet-7", "efficientnet-lite4-11", "inception-v3-12", "resnet101-v2-7", "resnet152-v2-7", "efficientnet-v2-l-18"]
    path_partitions = sys.argv[1]
    previous_path = os.getcwd()
    os.chdir("src/server_with_tls/scripts")
    make_build()

    for model_name in path:
        test_path = f"../../../models/{model_name}/test_data_set_0/input_0.pb"
        inference_operators = run_inference(f"../../../models/{model_name}/{path_partitions}", test_path)
        inference_whole = run_inference(f"../../../models/{model_name}/", test_path)

        if inference_operators != inference_whole:
            print(f"Operators: {inference_operators}")
            print(f"Whole: {inference_whole}")
            exit(1)
        print()

    run_command("make clean")
    os.chdir(previous_path)

if __name__ == "__main__":
    main()