import re
import os
import sys
import subprocess
import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python3 generate_cache_stats.py <number_of_runs>")
    exit(1)

number_of_runs = int(sys.argv[1])
path = ["squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", "efficientnet-lite4-11/", "inception-v3-12/", "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2-l-18/"]
average = {'cycles': [], 'instructions': [], 'IPC': []}

def generate_stats():
    for i in range(len(path)):
        current_cache_info = {'cycles': [], 'instructions': []}
        for j in range(number_of_runs):
            record_command = f"sudo perf record -e cycles,instructions ./standalone_inference ../../../models/{path[i]} ../../../models/{path[i]}test_data_set_0/input_0.pb"
            print(record_command)
            output = subprocess.run(record_command, shell=True)
            if output.returncode != 0:
                print("Error: perf record failed.")
                exit(1)

            report_command = "sudo perf report --stdio"
            report_process = subprocess.Popen(report_command, shell=True, stdout=subprocess.PIPE)
            (analysis, err) = report_process.communicate()
            if err:
                print(f"Error: {err}")
                exit(1)

            analysis = analysis.decode('utf-8')

            match_cycles = re.search(r'of event \'cycles\'\n# Event count \(approx\.\): (\d+)', analysis)
            if match_cycles:
                cycles = int(match_cycles.group(1))
            else:
                print("Event count of match_cycles not found.")
                exit(1)

            match_instructions = re.search(r'of event \'instructions\'\n# Event count \(approx\.\): (\d+)', analysis)
            if match_instructions:
                instructions = int(match_instructions.group(1))
            else:
                print("Event count of match_instructions not found.")
                exit(1)

            remove_command = subprocess.run("sudo rm perf.data", shell=True)
            if remove_command.returncode != 0:
                print("Error: perf.data not removed.")
                exit(1)

            current_cache_info['cycles'].append(cycles)
            current_cache_info['instructions'].append(instructions)
        average['cycles'].append(int(sum(current_cache_info['cycles']) // number_of_runs))
        average['instructions'].append(int(sum(current_cache_info['instructions']) // number_of_runs))
        average['IPC'].append(average['instructions'][i] / average['cycles'][i])
    return average

def generate_csv(average):
    model_names = [
        "SqueezeNet 1.0", "MobileNet V2", "DenseNet121", "EfficientNet Lite4", 
        "Inception V3", "ResNet101 V2", "ResNet152 V2", "EfficientNet V2"
    ]
    
    rows = []
    for i in range(len(path)):
        rows.append({
            'Model': model_names[i],
            'IPC': f"{average['IPC'][i]:.2f}"
        })

    df = pd.DataFrame(rows)
    if not os.path.exists("results/"):
        os.mkdir("results")
    df.to_csv(f'results/table3.csv', index=False)

if __name__ == "__main__":
    current_path = os.getcwd()
    os.chdir(f"src/server_with_tls/scripts")
    os.system(f"make clean && make USE_CACHE_STATS={number_of_runs}")

    average = generate_stats()
    os.system(f"make clean")
    os.chdir(current_path)

    generate_csv(average)