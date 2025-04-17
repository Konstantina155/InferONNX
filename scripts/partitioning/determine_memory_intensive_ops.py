from collections import defaultdict
import sys
import os

LIMIT = 12.0

def parse_execution_times_operators(file_content):
    operators = {}
    lines = file_content.splitlines()
    current_time = 0.0
    first = True
    i = 0
    expected_step = 0
    densenet_exceptions = ["conv4_10/x2_1", "conv5_7/x1/bn_3.low", "conv5_7/x2/bn_2", "conv5_7/x2/bn_3.low", "conv5_7/x2_1"]

    for line in lines:
        if "Running step" in line:
            parts = line.split('"')
            step_info = int((parts[0].split(",")[0]).split("Running step")[1].strip())
            
            if (step_info == 0 and "Source" in parts[2]) or (parts[1] in densenet_exceptions) or ("concat" in parts[1] and "Source" in parts[2]):
                expected_step = 0
                if not first:
                    if current_operator == "im2col-adhoc":
                        current_operator = current_operator + str(i)
                        i += 1
                    operators[current_operator] = current_time
                first = False
                current_time = 0.0
            else:
                expected_step += 1

            current_operator = parts[1]
        
        elif "takes" in line:
            time_taken = float(line.split("takes")[1].split("ms")[0].strip())
            current_time += time_taken
        else:
            continue
    if current_operator == "im2col-adhoc":
        current_operator = current_operator + str(i)
        i += 1
    operators[current_operator] = current_time

    result = [{"operator": op, "total_time_ms": time} for op, time in operators.items()]
    return result

def parse_execution_times(file_content):
    operators = {}
    lines = file_content.splitlines()
    previous_operator = None
    current_operator = None
    current_time = 0.0
    expected_step = 0
    begin_embed = False

    for line in lines:
        if "Running step" in line:
            parts = line.split('"')
            step_info = int((parts[0].split(",")[0]).split("Running step")[1].strip())
            
            if len(parts) > 1:
                if step_info != expected_step:
                    begin_embed = True

                if begin_embed:
                    current_operator = previous_operator
                else:
                    current_operator = parts[1]
                    expected_step += 1
                    current_time = 0.0

                if parts[1] == 'im2col-adhoc' and parts[2] == ' Reshape':
                    begin_embed = False
                
                if current_operator != previous_operator and current_operator:
                    operators[current_operator] = current_time

                previous_operator = current_operator

        elif "takes" in line:
            time_taken = float(line.split("takes")[1].split("ms")[0].strip())
            current_time += time_taken
            operators[current_operator] = current_time
        else:
            continue

    result = [{"operator": op, "total_time_ms": time} for op, time in operators.items()]
    return result

def find_similar_keys(config, flag):
    grouped_keys = defaultdict(list)
    for key in config.keys():
        if '.' in key:
            dot_index = key.rfind('.')
            slash_index = key.rfind('/', dot_index)
            
            if slash_index != -1:  # Slash exists after the dot
                base_key = key
            else:
                base_key = key[:dot_index]
            grouped_keys[base_key].append(key)
        else:
            grouped_keys[key].append(key)

    for base_key, related_keys in grouped_keys.items():
        if len(related_keys) > 1:
            combined_value = sum(config[key] for key in related_keys)
            if flag == True:
                final_key = related_keys[1]
            else:
                final_key = related_keys[0]
            config[final_key] = combined_value
            for key in related_keys:
                if key != final_key:
                    config.pop(key)
    return config

def find_keys(config):
    previous_key = None
    keys_to_remove = []

    for key in config.keys():
        if 'im2col-adhoc' in key:
            config[previous_key] = config.get(previous_key, 0) + config[key]
            keys_to_remove.append(key)
        previous_key = key

    for key in keys_to_remove:
        config.pop(key)
    
    return config

def update_operator_times(one_config, other_config, model):
    if "_operators.txt" not in model:
        other_list = list(other_config.keys())
        other_config[other_list[1]] = other_config.get(other_list[1], 0) + other_config.pop(other_list[0])

    one_list = list(one_config.keys())
    one_config[one_list[1]] = one_config.get(one_list[1], 0) + one_config.pop(one_list[0])          

    flag = False
    if model == "densenet_operators":
        flag = True
    one_config = find_similar_keys(one_config, flag)
    other_config = find_similar_keys(other_config, False)

    return one_config, other_config

def check_ops(first, second):    
    if len(first) != len(second):
        print("Different number of operators")
        return False
    
    all_operators = set(first.keys()).intersection(set(second.keys()))

    if not all_operators:
        print("No common operators found")
        return False
    return True

def calculate_overhead(first, second):
    all_operators = set(first.keys()).intersection(set(second.keys()))
    overhead = {}
    for operator in all_operators:
        time_in_first = first[operator]
        time_in_second = second[operator]
        overhead[operator] = (time_in_second - time_in_first) / time_in_first
    return overhead
 
def heaviest_operators(overhead):
    filtered_times = {k: v for k, v in overhead.items() if v > LIMIT}
    sorted_times = sorted(filtered_times, key=lambda x: filtered_times[x], reverse=True)
    return sorted_times

if __name__ == "__main__":
    directory = 'memory_intensive_ops'
    output_file_path = directory + '/operator_overhead.txt'

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        if os.path.isfile(file):
            with open(file, 'r') as f:
                file_content = f.read()

            if "_operators.txt" in file:
                func2 = parse_execution_times_operators
                label = "SGX-Files"
            else:
                func2 = parse_execution_times
                label = "SGX"
            func1 = parse_execution_times
            sections = file_content.split(label)

            systems = []
            for idx, section in enumerate(sections):
                if idx == 0:
                    exec = func1(section)
                else:
                    exec = func2(section)
                systems.append(exec)
            first = {entry["operator"]: entry["total_time_ms"] for entry in systems[0]}
            second = {entry["operator"]: entry["total_time_ms"] for entry in systems[1]}

            if "_operators.txt" in file:
                first = find_keys(first)
                second = find_keys(second)

            first, second = update_operator_times(first, second, file[:-4]) 

            if check_ops(first, second) == False:
                exit(1)

            if "_operators.txt" in file:
                overhead = calculate_overhead(first, second)
            else:
                overhead = calculate_overhead(first, second)
            heaviest = heaviest_operators(overhead)
            with open(output_file_path, "a") as out_file:
                if heaviest:
                    out_file.write(f"Heaviest operators for {os.path.basename(file).capitalize()[:-4]}:\n")
                for operator in heaviest:
                    if overhead[operator] > LIMIT:
                        out_file.write(f"{operator}: {overhead[operator]:.2f}x\n")
                out_file.write("\n")