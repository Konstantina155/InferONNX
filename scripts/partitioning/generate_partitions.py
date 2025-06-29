import os
import sys
import re
import onnx
import shutil
import subprocess
import time

class Partition:
    def __init__(self):
        self.index = 0

    def increment_index(self):
        self.index += 1

    def get_index(self):
        return self.index

### HELPER FUNCTIONS ###
def run_command(command):
    try:
        output = subprocess.Popen(command, shell=True)
        output.wait() 
    except subprocess.CalledProcessError as e:
        print(f"Command {command} failed with error: {e}")
        raise

def run_command_with_output(cmd, cwd=None):
    output = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out_stdout, out_stderr = output.communicate()
    if output.returncode != 0:
        raise Exception(f"Command {cmd} failed with error: {out_stderr.strip()}")
    return out_stdout.decode('utf-8')

def make_build():
    run_command("make clean")
    run_command("make")

def version_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_sorted_files_reversed(directory):
    try:
        files = os.listdir(directory)
        files = [f for f in files if os.path.isfile(os.path.join(directory, f))]

        files.sort(key=version_key, reverse=True)

        return [os.path.join(directory, f) for f in files]
    except Exception as e:
        print("Error sorting files:", e)
        return []

def clean_name(path):
    base_name = os.path.basename(path)
    return base_name.rsplit(".", 1)[0] if "." in base_name else base_name

def run_inference(directory, test_path):
    command = f"src/server_with_tls/scripts/./standalone_inference {directory} {test_path}"
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

def reverse_number(directory):
    files = [f for f in os.listdir(directory) if re.match(r".*_split\d+\.onnx$", f)]
    file_map = {
        int(re.search(r"_split(\d+)\.onnx", f).group(1)): f
        for f in files
    }

    if not file_map:
        print("No matching files found.")
        return

    max_num = max(file_map)
    mid = max_num // 2

    for old_num in range(mid + 1):
        new_num = max_num - old_num
        if old_num == new_num:
            continue

        old_file = file_map[old_num]
        new_file = file_map[new_num]

        old_path = os.path.join(directory, old_file)
        new_path = os.path.join(directory, new_file)
        temp_path = os.path.join(directory, "temp_swap.onnx")

        shutil.copy2(old_path, temp_path)
        os.rename(new_path, old_path)
        os.rename(temp_path, new_path)

        print(f"Swapped: {old_file} <-> {new_file}")
    print()
    
## extract the input/output operator names
def extract_input_ops(file):
    model = onnx.load(file)
    input_nodes = {node.name for node in model.graph.input}
    initializer_nodes = {node.name for node in model.graph.initializer}
    return input_nodes - initializer_nodes

def extract_output_ops(file):
    model = onnx.load(file)
    return {node.name for node in model.graph.output}

### MAIN FUNCTIONS ###
def heavy_operator_list():
    models = {}
    current_model = None

    with open("memory_intensive_ops/operator_overhead.txt", "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("Heaviest operators for"):
            current_model = line.split(":", 1)[0].split("for", 1)[1].strip().lower()
            models[current_model] = set()
        elif current_model:
            op_name = line.split(":", 1)[0].strip()
            if op_name:
                models[current_model].add(clean_name(op_name))
    return models

def calc_peak_memory_usage(file, inputs):
    command = f"python3 scripts/utils/peak_memory_usage.py -f {os.path.basename(file)[:-5]}.out -m {previous_path}/{file} -i {inputs}"
    output = run_command_with_output(command)
    
    peak_memory_usage_line = next(line for line in output.split("\n") if "Peak memory usage" in line)
    if peak_memory_usage_line:
        return round(float(peak_memory_usage_line.split(" ")[-2]), 2)
    else:
        raise ValueError(f"Peak memory usage not found when running for file: {file} with input(s): {inputs}.")

## find the input name -> '.pb' files
def find_model_inputs_filename(file):
    output = run_command_with_output(f"python3 scripts/utils/protobuf_write_file.py {file}")
    lines = output.split("\n")

    if not lines or lines == ['']:
        raise ValueError(f"No input filenames found for file: {file}")
    return " ".join(lines)

def union_models(model_name, partition_obj, models_to_union):
    first_model = models_to_union[0]
    last_model = models_to_union[-1]
    models_to_union_str = " ".join(models_to_union)
    merged_model_name = f"models/{model_name}/new_partitions/{model_name}_split{partition_obj.get_index()}.onnx"
    
    print(f"Merging models from {first_model} to {last_model} into {merged_model_name}")
    partition_obj.increment_index()
    
    run_command(f"python3 scripts/partitioning/union_ONNX_files.py {models_to_union_str} {merged_model_name}")
    return merged_model_name

def check_if_partitioning_small(files):
    total_memory_usage = 0
    for file in files:
        inputs_filename = find_model_inputs_filename(file)
        total_memory_usage += calc_peak_memory_usage(file, inputs_filename)
    print("Total peak memory usage: ", round(total_memory_usage, 2))
    return total_memory_usage, total_memory_usage < EPC_SIZE

def add_fill_inputs(fill_inputs, inputs, current_operator, last_operator, primary_inputs):
    if current_operator == last_operator or not inputs:
        return
    fill_inputs.update(input for input in inputs if input not in primary_inputs)

def clean_fill_inputs(fill_inputs, models_to_union, outputs, mode=None):
    if not fill_inputs:
        return
    
    if mode == "heavy":
        fill_inputs.difference_update(outputs)
    else:
        all_outputs = set()
        for op in models_to_union:
            all_outputs.update(extract_output_ops(op))
        fill_inputs.difference_update(all_outputs)

def not_union(partition_obj, operator, model_name):
    output_path = f"models/{model_name}/new_partitions/{model_name}_split{partition_obj.get_index()}.onnx"
    run_command(f"cp {operator} {output_path}")
    partition_obj.increment_index()

def find_input_names(operators):
    for operator in reversed(operators):
        inputs = extract_input_ops(operator)
        if inputs:
            return inputs
    return set()

def check_empty_union(models_to_union, model_name, partition_obj):
    if not models_to_union:
        return None   

    first_model = models_to_union[0]
    outputs = extract_output_ops(first_model)
    if len(models_to_union) == 1:
        print(f"Partition model FILL [cp previous] {first_model} ->> {model_name}_split{partition_obj.get_index()}.onnx")
        not_union(partition_obj, first_model, model_name)
        return extract_input_ops(first_model), outputs
    
    last_model = models_to_union[-1]
    print(f"Partition model FILL from operator {outputs} to [file_name]: {last_model}")
    if DEBUG:
        print(f"Partition model FILL from operator {outputs} to operator: {extract_output_ops(last_model)}")
    merged_model = union_models(model_name, partition_obj, models_to_union)
    return extract_input_ops(merged_model), extract_output_ops(merged_model)

def find_embedded_fill_inputs(models_to_union2, input_set, output_set, start_idx, end_idx, model_name, partition_obj):
    for idx in range(start_idx + 1, end_idx):
        output_names_set = extract_output_ops(models_to_union2[idx])
        print("Output name set: ", output_names_set)

        if input_set & output_names_set:
            sub_models_to_union = models_to_union2[start_idx:idx]
            if start_idx == idx:
                sub_models_to_union = [models_to_union2[idx]]
            print(f"Sub-models to union: {sub_models_to_union}, start_idx: {start_idx}, idx: {idx}")
            
            small_input_set, small_output_set = check_empty_union(sub_models_to_union, model_name, partition_obj)
            input_set.update(small_input_set)
            output_set.update(small_output_set)
            input_set.difference_update(output_names_set)
            print(f"Input set updated: {input_set}, start_idx: {start_idx}")
            
            start_idx = idx
            print(f"Start_idx: {start_idx}")

    if start_idx < end_idx:
        sub_models_to_union = models_to_union2[start_idx:end_idx]
    elif start_idx == end_idx:
        sub_models_to_union = [models_to_union2[end_idx]]

    small_input_set, small_output_set = check_empty_union(sub_models_to_union, model_name, partition_obj)
    input_set.update(small_input_set)
    output_set.update(small_output_set)

    return start_idx

def fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj):
    have_to_fill = False
    indexes = []
    models_to_union2 = models_to_union

    print("check fill in: ", fill_inputs, models_to_union[-1])

    for fill in fill_inputs:
        for idx2, operator2 in enumerate(models_to_union2):
            outputs1 = extract_output_ops(operator2)
            if fill in outputs1 and extract_input_ops(operator2):
                have_to_fill = True
                print(f"Remove fill\nFill inputs HAVE TO {fill} for operator:", outputs1)
                indexes.append(idx2)
                break

    if not have_to_fill:
        return False, inputs
    
    indexes.sort()
    input_set, output_set = set(), set()

    start_idx = indexes[0]
    for i in range(1, len(indexes)):
        end_idx = indexes[i]
        print(f"Processing models between {start_idx} and {end_idx}.")

        start_idx = find_embedded_fill_inputs(models_to_union2, input_set, output_set, start_idx, end_idx, model_name, partition_obj)

    if indexes[-1] <= len(models_to_union2) - 1:
        start_idx = indexes[-1]
        end_idx = len(models_to_union2)
        print(f"Final range: {start_idx} to {end_idx}")
        start_idx = find_embedded_fill_inputs(models_to_union2, input_set, output_set, start_idx, end_idx, model_name, partition_obj)

    print("Final Input and Output sets: ", input_set, output_set)
    inputs1 = input_set - output_set
    print("Inputs1 after fill: ", inputs1)

    if not inputs1:
        return True, inputs
    return True, inputs1

def process_single_model(previous_model, model_name, partition_obj):
    print(f"Partition model [cp previous] {previous_model} ->> {model_name}_split{partition_obj.get_index()}.onnx")
    not_union(partition_obj, previous_model, model_name)
    inputs = extract_input_ops(previous_model)
    return inputs

def process_union_models(models_to_union, fill_inputs, inputs, model_name, partition_obj):
    have_to_fill, inputs = fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj)
    if not have_to_fill:
        merged_model = union_models(model_name, partition_obj, models_to_union)
        print("Partition model from [file_name] ", models_to_union[0], " to [file_name] ", models_to_union[-1]) 
        if DEBUG:
            print("Partition model from operator ", extract_output_ops(models_to_union[0]), " to operator ", extract_input_ops(models_to_union[-1])) 
        inputs = extract_input_ops(merged_model)
    return inputs

def update_clean_inputs(fill_inputs, inputs, models_to_union, outputs, operator, last_operator, primary_inputs, mode=""):
    clean_fill_inputs(fill_inputs, models_to_union, outputs, mode)
    add_fill_inputs(fill_inputs, inputs, operator, last_operator, primary_inputs)

## partition small models by splitting the model in half
def partition_small_model(count_whole, operators, model_name, heavy_ops_set):
    print("Partition small model")
    total_memory_usage = 0
    partition_obj = Partition()
    fill_inputs = set()
    models_to_union = []

    primary_inputs = find_input_names(operators)
    print(primary_inputs)

    last_operator = operators[-1]
    for operator in operators:    
        inputs_filename = find_model_inputs_filename(operator)
        peak_memory_usage = calc_peak_memory_usage(operator, inputs_filename)
        total_memory_usage += peak_memory_usage

        inputs = extract_input_ops(operator)
        outputs = extract_output_ops(operator)
        print("FILL:", fill_inputs)

        standalone_partition = False
        if any(output in heavy_ops_set for output in outputs):
                standalone_partition = True
                print(f"Memory usage before heavy-weight: {round(total_memory_usage-peak_memory_usage, 2)}, after: {round(peak_memory_usage, 2)} for file {operator}")

                if operator != operators[0] and models_to_union:
                    if len(models_to_union) == 1:
                        first_model = models_to_union[0]
                        inputs1 = process_single_model(first_model, model_name, partition_obj)
                        operator_to_add = first_model
                    else:
                        inputs1 = process_union_models(models_to_union, fill_inputs, inputs, model_name, partition_obj)
                        operator_to_add = operator
                    update_clean_inputs(fill_inputs, inputs1, models_to_union, outputs, operator_to_add, last_operator, primary_inputs, mode="")
                
                index = partition_obj.get_index()
                not_union(partition_obj, operator, model_name)
                update_clean_inputs(fill_inputs, inputs, models_to_union, outputs, operator, last_operator, primary_inputs, mode="heavy")
                print(f"Partition model for heavy-weight from operator {operator} ->> {model_name}_split{index}.onnx")
                print(f"Fill inputs heavy {fill_inputs} for operator:", outputs)
                models_to_union = []
                    
                continue
        if total_memory_usage > count_whole/2:
            print(f"Memory usage before EPC, until {round(count_whole/2,2)}: ", round(total_memory_usage - peak_memory_usage, 2), " for file ", operator)
            total_memory_usage = peak_memory_usage

            if not models_to_union:
                print("Models to union is None!")
                break

            if len(models_to_union) == 1:
                first_model = models_to_union[0]
                inputs1 = process_single_model(first_model, model_name, partition_obj)
                operator_to_add = first_model
            else:
                inputs1 = process_union_models(models_to_union, fill_inputs, inputs, model_name, partition_obj)
                operator_to_add = operator
            
            update_clean_inputs(fill_inputs, inputs1, models_to_union, outputs, operator_to_add, last_operator, primary_inputs, mode="")
            models_to_union = [operator]
            print(f"Fill inputs epc {fill_inputs} for operator:", outputs)

        if operator not in models_to_union and not standalone_partition: models_to_union.append(operator)

    print("Total peak memory usage: ", round(total_memory_usage, 2))
    if models_to_union:
        first_model = models_to_union[0]
        if first_model != operators[-1]:
            inputs = process_union_models(models_to_union, fill_inputs, inputs, model_name, partition_obj)
            outputs = extract_output_ops(first_model)
        else:
            print(f"Partition model from operator [last partition - cp] {operator} ->> {model_name}_split{partition_obj.get_index()}.onnx")
            not_union(partition_obj, operator, model_name)
        clean_fill_inputs(fill_inputs, models_to_union, outputs)
    print("Final fill inputs: ", fill_inputs)

def partition_model(heavy_ops_set, operators, model_name):
    if model_name == "squeezenet1.0-7":
        total_memory_usage, small_model = check_if_partitioning_small(operators)
        if small_model:
            partition_small_model(total_memory_usage, operators, model_name, heavy_ops_set)
            return
    
    current_memory_usage = 0
    partition_obj = Partition()
    fill_inputs = set()
    models_to_union = []

    primary_inputs = find_input_names(operators)
    print(primary_inputs)

    last_operator = operators[-1]
    for operator in operators:
        inputs_filename = find_model_inputs_filename(operator)
        peak_memory_usage = calc_peak_memory_usage(operator, inputs_filename)
        current_memory_usage += peak_memory_usage

        inputs = extract_input_ops(operator)
        outputs = extract_output_ops(operator)
        print("FILL:", fill_inputs)

        standalone_partition = False
        if any(output in heavy_ops_set for output in outputs) or peak_memory_usage > EPC_SIZE:
                standalone_partition = True
                print(f"Memory usage before heavy-weight: {round(current_memory_usage-peak_memory_usage, 2)}, after: {round(peak_memory_usage, 2)} for file {operator}")
                current_memory_usage = 0

                if operator != operators[0] and models_to_union:
                    if len(models_to_union) == 1:
                        first_model = models_to_union[0]
                        inputs1 = process_single_model(first_model, model_name, partition_obj)
                        operator_to_add = first_model
                    else:
                        inputs1 = process_union_models(models_to_union, fill_inputs, inputs, model_name, partition_obj)
                        operator_to_add = operator
                    update_clean_inputs(fill_inputs, inputs1, models_to_union, outputs, operator_to_add, last_operator, primary_inputs, mode="")

                index = partition_obj.get_index()
                not_union(partition_obj, operator, model_name)
                update_clean_inputs(fill_inputs, inputs, models_to_union, outputs, operator, last_operator, primary_inputs, mode="heavy")
                print(f"Partition model for heavy-weight from operator {operator} ->> {model_name}_split{index}.onnx")
                print(f"Fill inputs heavy {fill_inputs} for operator:", outputs)
                models_to_union = []
                    
                continue
        if current_memory_usage > EPC_SIZE:
            print(f"Memory usage before EPC, until {round(current_memory_usage-peak_memory_usage, 2)}, for file ", operator)
            current_memory_usage = peak_memory_usage

            if not models_to_union:
                print("Models to union is None!")
                break

            if len(models_to_union) == 1:
                first_model = models_to_union[0]
                inputs1 = process_single_model(first_model, model_name, partition_obj)
                operator_to_add = first_model
            else:
                inputs1 = process_union_models(models_to_union, fill_inputs, inputs, model_name, partition_obj)
                operator_to_add = operator
            
            update_clean_inputs(fill_inputs, inputs1, models_to_union, outputs, operator_to_add, last_operator, primary_inputs, mode="")
            models_to_union = [operator]
            print(f"Fill inputs epc {fill_inputs} for operator:", outputs)
            
        if operator not in models_to_union and not standalone_partition: models_to_union.append(operator)
    
    print("Remaining memory usage: ", round(current_memory_usage, 2))
    if models_to_union:
        first_model = models_to_union[0]
        if first_model != operators[-1]:
            inputs = process_union_models(models_to_union, fill_inputs, inputs, model_name, partition_obj)
            outputs = extract_output_ops(first_model)
        else:
            print(f"Partition model from operator [last partition - cp] {operator} ->> {model_name}_split{partition_obj.get_index()}.onnx")
            not_union(partition_obj, operator, model_name)
        clean_fill_inputs(fill_inputs, models_to_union, outputs)
    print("Final fill inputs: ", fill_inputs)

def main():  
    global EPC_SIZE, previous_path, DEBUG
    DEBUG = False
    if len(sys.argv) == 2 and sys.argv[1] == "--debug":
        DEBUG = True
    EPC_SIZE = 85

    model_directories = ["squeezenet1.0-7" , "mobilenetv2-7", "efficientnet-lite4-11", "resnet101-v2-7", "resnet152-v2-7", "densenet-7", "inception-v3-12", "efficientnet-v2-l-18"]
    path_to_models = "models/"
    previous_path = os.getcwd()
    os.chdir('src/server_with_tls/scripts/')
    make_build()
    os.chdir(previous_path)
    os.makedirs("../input_files", exist_ok=True)

    heavy_operator_models = heavy_operator_list()
    for model, operators in heavy_operator_models.items():
        print(f"Model: {model}")
        print("Operators: ", operators)

    start = time.time()
    for model_name in model_directories:
        partition_dir = f"{path_to_models}{model_name}/new_partitions/"
        if os.path.exists(partition_dir):
            print(f"Directory {partition_dir} already exists")
            exit(1)
        os.makedirs(partition_dir)
        
        heavy_ops_set = heavy_operator_models.get(model_name, set())
        operators = get_sorted_files_reversed(f"{path_to_models}{model_name}/operators/")

        partition_model(heavy_ops_set, operators, model_name)
        
        test_input_path = f"{path_to_models}{model_name}/test_data_set_0/input_0.pb"
        reverse_number(partition_dir)
        inference_operators = run_inference(partition_dir, test_input_path)
        inference_whole = run_inference(f"{path_to_models}{model_name}/", test_input_path)

        if inference_operators != inference_whole:
            print(f"Operators: {inference_operators}")
            print(f"Whole: {inference_whole}")
            end = time.time()
            print(f"Took: {end - start} sec")
            exit(1)
        print()
    end = time.time()
    print(f"Took: {end - start} sec")

    run_command(f"rm -rf ../input_files/")
    os.chdir('src/server_with_tls/scripts/')
    run_command("make clean")
    os.chdir(previous_path)


if __name__ == "__main__":
    main()
