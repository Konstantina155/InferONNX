import os
import sys
import subprocess

if len(sys.argv) != 2:
    print("Usage: python3 check_accuracy.py <partitions_folder>")
    exit(1)

path = ["squeezenet1.0-7", "mobilenetv2-7", "densenet-7", "efficientnet-lite4-11", "inception-v3-12", "resnet101-v2-7", "resnet152-v2-7", "efficientnet-v2-l-18"]
path_partitions = sys.argv[1]

def run_inference(directory, test_path):
    command = f"./standalone_inference {directory} {test_path}"
    try:
        output = subprocess.run(command, shell=True, stderr=subprocess.PIPE, check=True)
        output = output.stderr.decode("utf-8")
        
        beginning_of_max = output.rfind("Max is")
        if beginning_of_max != -1:
            end_of_sentence = output.find("!", beginning_of_max)
            if end_of_sentence != -1:
                print(output[beginning_of_max:end_of_sentence + 1])

        return output[beginning_of_max:end_of_sentence + 1]
    except subprocess.CalledProcessError as error:
        print(error)
        exit(1)

if __name__ == "__main__":
    previous_path = os.getcwd()
    os.chdir("src/server_with_tls/scripts")
    os.system("make clean && make")


    for model_name in path:
        test_path = f"../../../models/{model_name}/test_data_set_0/input_0.pb"
        inference_operators = run_inference(f"../../../models/{model_name}/{path_partitions}", test_path)
        inference_whole = run_inference(f"../../../models/{model_name}/", test_path)

        if inference_operators != inference_whole:
            print(f"Operators: {inference_operators}")
            print(f"Whole: {inference_whole}")
            exit(1)
        print()

    os.system("make clean")
    os.chdir(previous_path)