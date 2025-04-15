import re
import os
import sys
import subprocess

if len(sys.argv) != 2:
    print("Usage: python3 generate_chache_stats.py")
    exit(1)

path = ["squeezenet1.0-7/", "mobilenetv2-7/", "densenet-7/", "efficientnet-lite4-11/", 
        "inception-v3-12/", "resnet101-v2-7/", "resnet152-v2-7/", "efficientnet-v2/"]
average = {'cache-references': [], 'cache-misses': [], 'cycles': [], 'instructions': [], 'IPC': [], 'cycles-per-instruction': []}

search_dirs = ["/hdd", os.path.expanduser("~")]
inferONNX_path = None
for search_dir in search_dirs:
    for root, dirs, files in os.walk(search_dir):
        if "InferONNX" in dirs:
            inferONNX_path = os.path.join(root, "InferONNX")
            print("Found InferONNX directory at:", os.path.join(root, "InferONNX"))
            break
if not inferONNX_path:
    print("inferONNX path cannot be found!")
    exit(1)

os.chdir(f"{inferONNX_path}/src/tls_support/scripts")
os.system(f"make clean && make USE_CACHE_STATS={num_runs}")

for i in range(len(path)):
    current_cache_info = {'cache-references': [], 'cache-misses': [], 'cycles': [], 'instructions': []}
    for j in range(num_runs):
        record_command = f"sudo perf record -e cache-references,cache-misses,cycles,instructions ./standalone_inference {inferONNX_path}/models/{path[i]} {inferONNX_path}/models/{path[i]}test_data_set_0/input_0.pb"
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

        match_references = re.search(r'of event \'cache-references\'\n# Event count \(approx\.\): (\d+)', analysis)
        if match_references:
            cache_references = int(match_references.group(1))
        else:
            print("Event count of match_references not found.")
            exit(1)

        match_misses = re.search(r'of event \'cache-misses\'\n# Event count \(approx\.\): (\d+)', analysis)
        if match_misses:
            cache_misses = int(match_misses.group(1))
        else:
            print("Event count of match_misses not found.")
            exit(1)

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

        current_cache_info['cache-references'].append(cache_references)
        current_cache_info['cache-misses'].append(cache_misses)
        current_cache_info['cycles'].append(cycles)
        current_cache_info['instructions'].append(instructions)
    average['cache-references'].append(int(sum(current_cache_info['cache-references']) // runs))
    average['cache-misses'].append(int(sum(current_cache_info['cache-misses']) // runs))
    average['cycles'].append(int(sum(current_cache_info['cycles']) // runs))
    average['instructions'].append(int(sum(current_cache_info['instructions']) // runs))
    average['IPC'].append(average['instructions'][i] / average['cycles'][i])
    average['cycles-per-instruction'].append(average['cycles'][i] / average['instructions'][i])

with open("cache_stats.txt", "a") as f:
    for i in range(len(path)):
        onnx_file = path[i][:-1] + ".onnx"
        f.write(f"Executing model {onnx_file[i]} for {runs} runs:")
        f.write(f"Average cache_references: {average['cache-references'][i]}")
        f.write(f"Average cache_misses: {average['cache-misses'][i]}")
        f.write(f"Average cycles: {average['cycles'][i]}")
        f.write(f"Average instructions: {average['instructions'][i]}")
        f.write(f"Average IPC: {average['IPC'][i]}")
        f.write(f"Average cycles-per-instruction: {average['cycles-per-instruction'][i]}")