import os
import subprocess
import ast
import re
import onnx

EXTRACT_NAME = -2

class Partition:
    def __init__(self):
        self.index = 0

    def increment_index(self):
        self.index += 1

    def get_index(self):
        return self.index

EPC_SIZE = 85

model_directory = ["squeezenet1.0-7"]#, "mobilenetv2-7", "efficientnet-lite4-11", "resnet101-v2-7", "resnet152-v2-7", "densenet-7", "inception-v3-12", "efficientnet-v2-l-18"]
path_to_models = "models/"

### HELPER FUNCTIONS ###
def get_sorted_files_reversed(directory):
    try:
        result = subprocess.run(f"ls -1 {directory} | sort -Vr", 
                                shell=True, 
                                check=True, 
                                text=True, 
                                capture_output=True)
        files = result.stdout.strip().split("\n")
        return [os.path.join(directory, file) for file in files if file and os.path.isfile(os.path.join(directory, file))]
    except subprocess.CalledProcessError as e:
        print("Error sorting files:", e)
        return []

def clean_name(path):
    base_name = os.path.basename(path)

    if "." in base_name:
        base_name = path.rsplit(".", 1)[0]
        return base_name
    else:
        return path
    
## extract the input/output operator names
def extract_inputs_ops(file):
    output = subprocess.run(f"python3 scripts/utils/onnx_get_io.py {file}", shell=True, check=True, text=True, capture_output=True)

    for line in output.stdout.split("\n"):
        if line.startswith("Inputs:"):
            return ast.literal_eval(line.split(":", 1)[1].strip())
    return []

def extract_outputs_ops(file):
    output = subprocess.run(f"python3 scripts/utils/onnx_get_io.py {file}", shell=True, check=True, text=True, capture_output=True)

    for line in output.stdout.split("\n"):
        if line.startswith("Outputs:"):
            return ast.literal_eval(line.split(":", 1)[1].strip())
    return []

### MAIN FUNCTIONS ###
def heavy_operator_list():
    models = {}
    model_pattern = re.compile(r"Heaviest operators for (.*?):")
    operator_pattern = re.compile(r"^([\w/.-]+):")

    with open("memory_intensive_ops/operator_overhead.txt", "r") as file:
        for line in file:
            model_match = model_pattern.match(line)
            operator_match = operator_pattern.match(line)

            if model_match:
                current_model = model_match.group(1).lower()
                models[current_model] = set()

            elif operator_match and current_model:
                operator_name = clean_name(operator_match.group(1))
                models[current_model].add(operator_name)
    return models

def calc_peak_memory_usage(file, inputs):
    command = f"python3 scripts/utils/peak_memory_usage.py -f {os.path.basename(file)[:-5]}.out -m dummy_folder/ -i {inputs}"
    output = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if "Error" in output.stderr.decode("utf-8"):
        print(output.stderr)
        exit(1)
    output = output.stdout.decode("utf-8")
    
    peak_memory_usage_line = next(line for line in output.split("\n") if "Peak memory usage" in line)
    peak_memory_usage = round(float(peak_memory_usage_line.split(" ")[-2]), 2)
    return peak_memory_usage

## find the input name -> '.pb' files
def find_model_inputs_filename(file):
    result = subprocess.run(f"python3 scripts/utils/protobuf_write_file.py {file}", shell=True, check=True, stdout=subprocess.PIPE)
    inputs_filename = " ".join(result.stdout.decode("utf-8").split("\n"))
    return inputs_filename

def union_models(model_name, partition_obj, models_to_union):
    models_to_union_str = " ".join(models_to_union)
    merged_model_name = f"models/{model_name}/new_partitions/{model_name}_split{partition_obj.get_index()}.onnx"
    command = f"python3 scripts/partitioning/union_ONNX_files.py {models_to_union_str} {merged_model_name}"
    print(f"Merging models from {models_to_union[0]} to {models_to_union[-1]} into {merged_model_name}")
    partition_obj.increment_index()
    os.system(command)
    return merged_model_name

def check_if_partitioning_small(files):
    total_memory_usage = 0
    for file in files:
        os.system(f"cp {file} dummy_folder/")
        inputs_filename = find_model_inputs_filename(file)
        total_memory_usage += calc_peak_memory_usage(file, inputs_filename)
        os.system("rm -f *.pb && rm -f dummy_folder/*")
    print("Total peak memory usage: ", round(total_memory_usage, 2))
    return total_memory_usage, total_memory_usage < EPC_SIZE

def add_fill_inputs(fill_inputs, inputs, current_operator, last_operator, primary_inputs):
    if current_operator == last_operator or not inputs:
        return

    for input in inputs:
        if input not in primary_inputs:
            fill_inputs.add(input)

def clean_fill_inputs(fill_inputs, models_to_union, outputs, mode=None):
    if not fill_inputs:
        return
    
    if mode == "heavy":
        for output in outputs:
            if output in fill_inputs:
                fill_inputs.remove(output)
        return

    for op in models_to_union:
        for output in extract_outputs_ops(op):
            if output in fill_inputs:
                fill_inputs.remove(output)

def not_union(partition_obj, operator, model_name):
    os.system(f"cp {operator} models/{model_name}/new_partitions/{model_name}_split{partition_obj.get_index()}.onnx")
    partition_obj.increment_index()

def find_input_names(operators):
    for operator in reversed(operators):
        inputs = extract_inputs_ops(operator)
        if inputs != []:
            return inputs
    return None

def check_empty_union(models_to_union, model_name, partition_obj):
    if len(models_to_union) == 0: return None
    if len(models_to_union) == 1:
        print(f"Partition model FILL [cp previous] {models_to_union[0]} ->> {model_name}_split{partition_obj.get_index()}.onnx")
        not_union(partition_obj, models_to_union[0], model_name)
        return extract_inputs_ops(models_to_union[0]), extract_outputs_ops(models_to_union[0])
    
    print("Partition model FILL from operator ", extract_outputs_ops(models_to_union[0]), " to operator ", extract_outputs_ops(models_to_union[len(models_to_union)-1]))
    merged_model = union_models(model_name, partition_obj, models_to_union)
    return extract_inputs_ops(merged_model), extract_outputs_ops(merged_model)

def fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj):
    have_to_fill = False
    inputs1 = None
    indexes = []
    models_to_union2 = models_to_union
    print("check fill in: ", fill_inputs, models_to_union[len(models_to_union) - 1])
    for fill in fill_inputs:
        for idx2, operator2 in enumerate(models_to_union2):
            outputs1 = extract_outputs_ops(operator2)
            if fill in outputs1 and extract_inputs_ops(operator2) != []:
                have_to_fill = True
                print("Remove fill")
                print(f"Fill inputs HAVE TO {fill} for operator:", outputs1)
                indexes.append(idx2)
                break
    if have_to_fill:
        indexes.sort()
        output_set = set()
        input_set = set()
        print(indexes, len(models_to_union2), models_to_union2[0], models_to_union2[-1])
        idx2 = 0
        for i in range(1, len(indexes)):
            start_idx = indexes[i-1] if idx2 == 0 else idx2
            end_idx = indexes[i]
            print(start_idx, end_idx)

            for idx in range(start_idx + 1, end_idx):
                model = onnx.load(models_to_union2[idx])
                output_names_set = set(model.graph.node[0].output)
                print("Output name set: ", output_names_set)

                if input_set & output_names_set:
                    sub_models_to_union = models_to_union2[start_idx:idx]
                    if start_idx == idx:
                        sub_models_to_union = [models_to_union2[idx]]
                    print("sub models to union: ", sub_models_to_union, start_idx, idx)
                    
                    input_list, output_list = check_empty_union(sub_models_to_union, model_name, partition_obj)
                    input_set.update(input_list)
                    print("Input set: ", input_set, start_idx)
                    input_set.difference_update(output_names_set)
                    print("Input set: ", input_set, start_idx)
                    output_set.update(output_list)
                    start_idx = idx
                    print("Input set: ", input_set, start_idx)
                    idx2 = idx

            if start_idx < end_idx:
                sub_models_to_union = models_to_union2[start_idx:end_idx]
            elif start_idx == end_idx:
                sub_models_to_union = [models_to_union2[end_idx]]

            input_list, output_list = check_empty_union(sub_models_to_union, model_name, partition_obj)
            input_set.update(input_list)
            output_set.update(output_list)
        if indexes[-1] <= len(models_to_union2) - 1:
            start_idx = indexes[-1]
            end_idx = len(models_to_union2)

            for idx in range(start_idx + 1, end_idx):
                model = onnx.load(models_to_union2[idx])
                output_names_set = set(model.graph.node[0].output)
                print("Output name set: ", output_names_set)

                if input_set & output_names_set:
                    sub_models_to_union = models_to_union2[start_idx:idx]
                    if start_idx == idx:
                        sub_models_to_union = [models_to_union2[idx]]
                    print("sub models to union: ", sub_models_to_union, start_idx, idx)
                    
                    input_list, output_list = check_empty_union(sub_models_to_union, model_name, partition_obj)
                    input_set.update(input_list)
                    input_set.difference_update(output_names_set)
                    output_set.update(output_list)
                    print("Input set: ", input_set, start_idx)
                    start_idx = idx

            if start_idx < end_idx:
                sub_models_to_union = models_to_union2[start_idx:end_idx]
            elif start_idx == end_idx:
                sub_models_to_union = [models_to_union2[end_idx]]

            input_list, output_list = check_empty_union(sub_models_to_union, model_name, partition_obj)
            input_set.update(input_list)
            output_set.update(output_list)

        print("Inputs1: ", input_set, output_set, models_to_union, indexes)
        inputs1 = list(input_set - output_set)
        print("input1: ", inputs1)
    if not inputs1: inputs1 = inputs
    return have_to_fill, inputs1

## partition small models by splitting the model in half
def partition_small_model(count_whole, operators, model_name, heavy_ops_set):
    print("Partition small model")
    total_memory_usage = 0
    partition_obj = Partition()
    fill_inputs = set()
    models_to_union = []

    primary_inputs = find_input_names(operators)
    print(primary_inputs)

    prev_operator = None
    last_operator = operators[-1]
    for operator in operators:
        os.system(f"cp {operator} dummy_folder/")
    
        inputs_filename = find_model_inputs_filename(operator)
        peak_memory_usage = calc_peak_memory_usage(operator, inputs_filename)
        current_memory_usage = peak_memory_usage
        total_memory_usage += peak_memory_usage

        print("FILL:", fill_inputs)
        inputs = extract_inputs_ops(operator)
        outputs = extract_outputs_ops(operator)

        standalone_partition = False
        for output in outputs:
            if output in heavy_ops_set:
                standalone_partition = True
                print("Memory usage before heavy-weight: ", round(current_memory_usage-peak_memory_usage, 2), " for file ", operator)
                print("Memory usage for heavy-weight: ", round(peak_memory_usage, 2), " for file ", operator)
                current_memory_usage = 0

                if prev_operator and models_to_union:
                    if len(models_to_union) == 1:
                        print(f"Partition model for heavy [cp previous] {models_to_union[0]} ->> {model_name}_split{partition_obj.get_index()}.onnx")
                        not_union(partition_obj, models_to_union[0], model_name)
                        operator2 = operators[-1]
                        inputs1 = extract_inputs_ops(models_to_union[0])
                    else:
                        have_to_fill, inputs1 = fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj)
                        if not have_to_fill:
                            merged_model = union_models(model_name, partition_obj, models_to_union)
                            print("Partition model for heavy-weight from operator ", extract_outputs_ops(models_to_union[0]), " to operator ", extract_outputs_ops(models_to_union[len(models_to_union)-1]))
                            inputs1 = extract_inputs_ops(merged_model)
                        operator2 = operator
                    clean_fill_inputs(fill_inputs, models_to_union, outputs)
                    add_fill_inputs(fill_inputs, inputs1, operator2, last_operator, primary_inputs)
                index = partition_obj.get_index()

                clean_fill_inputs(fill_inputs, models_to_union, outputs, "heavy")
                print(f"Partition model for heavy-weight from operator {operator} ->> {model_name}_split{index}.onnx")
                not_union(partition_obj, operator, model_name)
                add_fill_inputs(fill_inputs, inputs, operator, last_operator, primary_inputs)
                print(f"Fill inputs heavy {fill_inputs} for operator:", outputs)
                models_to_union = []
                    
                break
        if total_memory_usage > count_whole/2:
            print(f"Memory usage until {round(count_whole/2,2)}: ", round(total_memory_usage - peak_memory_usage, 2), " for file ", operator)
            total_memory_usage = peak_memory_usage

            if not models_to_union:
                print("Models to union is None!")
                break

            if len(models_to_union) == 1:
                print(f"Partition model for epc [cp previous] {models_to_union[0]} ->> {model_name}_split{partition_obj.get_index()}.onnx")
                not_union(partition_obj, models_to_union[0], model_name)
                operator2 = prev_operator
                inputs = extract_inputs_ops(models_to_union[0])
            else:
                have_to_fill, inputs = fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj)
                if not have_to_fill:
                    merged_model = union_models(model_name, partition_obj, models_to_union)
                    print("Partition model for epc from operator ", extract_outputs_ops(models_to_union[0]), " to operator ", extract_outputs_ops(models_to_union[len(models_to_union)-1])) 
                    inputs = extract_inputs_ops(merged_model)
                operator2 = operator
            clean_fill_inputs(fill_inputs, models_to_union, outputs)
            add_fill_inputs(fill_inputs, inputs, operator2, last_operator, primary_inputs)
            models_to_union = [operator]
            print(f"Fill inputs epc {fill_inputs} for operator:", outputs)

        os.system("rm -f *.pb && rm -f dummy_folder/*")
        prev_operator = operator
        if operator not in models_to_union and not standalone_partition: models_to_union.append(operator)

    print("Total peak memory usage: ", round(total_memory_usage, 2))
    if models_to_union:
        if models_to_union[0] != operators[len(operators) - 1]:
            have_to_fill, inputs = fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj)
            if not have_to_fill:
                merged_model = union_models(model_name, partition_obj, models_to_union)
                print("Partition model from operator [last partition] ", extract_outputs_ops(models_to_union[0]), " to operator ", extract_outputs_ops(models_to_union[len(models_to_union)-1])) 
                inputs = extract_inputs_ops(merged_model)
            outputs = extract_outputs_ops(models_to_union[0])
        else:
            print(f"Partition model from operator [last partition - cp] {operator} ->> {model_name}_split{partition_obj.get_index()}.onnx")
            not_union(partition_obj, operator, model_name)
        clean_fill_inputs(fill_inputs, models_to_union, outputs)
    print("Fill inputs: ", fill_inputs)

def partition_model(heavy_ops_set, operators, model_name):
    if not os.path.exists("dummy_folder"):
        os.system("mkdir dummy_folder")
    else:
        os.system(f"rm -rf dummy_folder/*")

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

    prev_operator = None
    last_operator = operators[-1]
    for operator in operators:
        os.system(f"cp {operator} dummy_folder/")

        inputs_filename = find_model_inputs_filename(operator)
        peak_memory_usage = calc_peak_memory_usage(operator, inputs_filename)
        current_memory_usage += peak_memory_usage

        print("FILL:", fill_inputs)
        inputs = extract_inputs_ops(operator)
        outputs = extract_outputs_ops(operator)

        standalone_partition = False
        for output in outputs:
            if output in heavy_ops_set:
                standalone_partition = True
                print("Memory usage before heavy-weight: ", round(current_memory_usage-peak_memory_usage, 2), " for file ", operator)
                print("Memory usage for heavy-weight: ", round(peak_memory_usage, 2), " for file ", operator)
                current_memory_usage = 0

                if prev_operator and models_to_union:
                    if len(models_to_union) == 1:
                        print(f"Partition model for heavy [cp previous] {models_to_union[0]} ->> {model_name}_split{partition_obj.get_index()}.onnx")
                        not_union(partition_obj, models_to_union[0], model_name)
                        operator2 = operators[-1]
                        inputs1 = extract_inputs_ops(models_to_union[0])
                    else:
                        have_to_fill, inputs1 = fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj)
                        if not have_to_fill:
                            merged_model = union_models(model_name, partition_obj, models_to_union)
                            print("Partition model for heavy-weight from operator ", extract_outputs_ops(models_to_union[0]), " to operator ", extract_outputs_ops(models_to_union[len(models_to_union)-1]))
                            inputs1 = extract_inputs_ops(merged_model)
                        operator2 = operator
                    clean_fill_inputs(fill_inputs, models_to_union, outputs)
                    add_fill_inputs(fill_inputs, inputs1, operator2, last_operator, primary_inputs)
                index = partition_obj.get_index()

                clean_fill_inputs(fill_inputs, models_to_union, outputs, "heavy")
                print(f"Partition model for heavy-weight from operator {operator} ->> {model_name}_split{index}.onnx")
                not_union(partition_obj, operator, model_name)
                total_memory_usage = 0
                add_fill_inputs(fill_inputs, inputs, operator, last_operator, primary_inputs)
                print(f"Fill inputs heavy {fill_inputs} for operator:", outputs)
                models_to_union = []
                    
                break
        if current_memory_usage > EPC_SIZE:
            print("Total peak memory usage: ", round(current_memory_usage-peak_memory_usage, 2))
            print("Operator: ", operator, " exceeds EPC size")

            if not models_to_union:
                print("Models to union is None!")
                break

            current_memory_usage = peak_memory_usage

            if len(models_to_union) == 1:
                print(f"Partition model for epc [cp previous] {models_to_union[0]} ->> {model_name}_split{partition_obj.get_index()}.onnx")
                not_union(partition_obj, models_to_union[0], model_name)
                operator2 = prev_operator
                inputs1 = extract_inputs_ops(models_to_union[0])
            else:
                have_to_fill, inputs1 = fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj)
                if not have_to_fill:
                    merged_model = union_models(model_name, partition_obj, models_to_union)
                    print("Partition model for epc from operator ", extract_outputs_ops(models_to_union[0]), " to operator ", extract_outputs_ops(models_to_union[len(models_to_union)-1])) 
                    inputs1 = extract_inputs_ops(merged_model)
                operator2 = operator
            clean_fill_inputs(fill_inputs, models_to_union, outputs)
            add_fill_inputs(fill_inputs, inputs1, operator2, last_operator, primary_inputs)
            models_to_union = [operator]
            print(f"Fill inputs epc {fill_inputs} for operator:", outputs)
            

        os.system("rm -f *.pb && rm -f dummy_folder/*")
        prev_operator = operator
        if operator not in models_to_union and not standalone_partition: models_to_union.append(operator)
    
    print("Remaining memory usage: ", round(current_memory_usage, 2))
    if models_to_union:
        if models_to_union[0] != operators[len(operators) - 1]:
            have_to_fill, inputs = fill_nodes(fill_inputs, inputs, models_to_union, model_name, partition_obj)
            if not have_to_fill:
                merged_model = union_models(model_name, partition_obj, models_to_union)
                print("Partition model from operator [last partition] ", extract_outputs_ops(models_to_union[0]), " to operator ", extract_outputs_ops(models_to_union[len(models_to_union)-1])) 
                inputs = extract_inputs_ops(merged_model)
            outputs = extract_outputs_ops(models_to_union[0])
        else:
            print(f"Partition model from operator [last partition - cp] {operator} ->> {model_name}_split{partition_obj.get_index()}.onnx")
            not_union(partition_obj, operator, model_name)
        clean_fill_inputs(fill_inputs, models_to_union, outputs)
    print("Fill inputs: ", fill_inputs)

    os.system("rm -rf dummy_folder/")

def run_inference(directory, test_path):
    command = f"src/server_with_tls/scripts/./standalone_inference {directory} {test_path}"
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

def reverse_number(directory):
    files = [f for f in os.listdir(directory) if re.match(r"^.*_split[0-9]+\.onnx$", f)]
    split_numbers = [int(re.search(r"_split([0-9]+)\.onnx", f).group(1)) for f in files]
    
    if not split_numbers:
        print("No matching files found.")
        return
    
    max_num = max(split_numbers)
    mid = max_num // 2
    
    for old_num in split_numbers:
        if old_num > mid:
            continue
        
        new_num = max_num - old_num
        old_file = next(f for f in files if f"_split{old_num}.onnx" in f)
        new_file = next(f for f in files if f"_split{new_num}.onnx" in f)
        
        old_path = os.path.join(directory, old_file)
        new_path = os.path.join(directory, new_file)
        temp_path = os.path.join(directory, "temp_swap.onnx")
        
        os.system(f"cp {old_path} {temp_path}")
        os.rename(new_path, old_path)
        os.rename(temp_path, new_path)        
        print(f"Swapped: {old_file} <-> {new_file}")
    print("\n")

def main():        
    previous_path = os.getcwd()
    os.chdir('src/server_with_tls/scripts/')
    os.system("make clean && make")
    os.chdir(previous_path)

    heavy_operator_models = heavy_operator_list()
    for model, operators in heavy_operator_models.items():
        print(f"Model: {model}")
        print("Operators: ", operators)

    for model_name in model_directory:
        partition_dir = f"{path_to_models}{model_name}/new_partitions/"
        if not os.path.exists(partition_dir):
            os.system(f"mkdir {partition_dir}")
        else:
            print(f"Directory {partition_dir} already exists")
            exit(1)

        heavy_ops_set = heavy_operator_models.get(model_name, set())
        operators = get_sorted_files_reversed(path_to_models + model_name + "/operators/")
        partition_model(heavy_ops_set, operators, model_name)

        test_path = f"{path_to_models}{model_name}/test_data_set_0/input_0.pb"
        reverse_number(partition_dir)
        inference_operators = run_inference(partition_dir, test_path)
        inference_whole = run_inference(f"{path_to_models}{model_name}/", test_path)

        if inference_operators != inference_whole:
            print(f"Operators: {inference_operators}")
            print(f"Whole: {inference_whole}")
            exit(1)
        print()

    os.chdir('src/server_with_tls/scripts/')
    os.system("make clean")
    os.chdir(previous_path)


if __name__ == "__main__":
    main()