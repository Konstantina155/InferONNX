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

def update_operator_times(config1, config2, model):
    if "_operators.txt" not in model:
        keys2 = list(config2.keys())
        config2[keys2[1]] = config2.get(keys2[1], 0) + config2.pop(keys2[0])

    keys1 = list(config1.keys())
    config1[keys1[1]] = config1.get(keys1[1], 0) + config1.pop(keys1[0])          

    flag = False
    if model == "densenet_operators":
        flag = True
    config1 = find_similar_keys(config1, flag)
    config2 = find_similar_keys(config2, False)

    return config1, config2

def check_ops(s1, s2):    
    if len(s1) != len(s2):
        print("Different number of operators")
        return False
    
    all_operators = set(s1.keys()).intersection(set(s2.keys()))

    if not all_operators:
        print("No common operators found")
        return False
    return True

def calculate_overhead(s1, s2):
    all_operators = set(s1.keys()).intersection(set(s2.keys()))
    overhead = {}
    for operator in all_operators:
        time_in_s1 = s1[operator]
        time_in_s2 = s2[operator]
        overhead[operator] = (time_in_s2 - time_in_s1) / time_in_s1
    return overhead
 
def heaviest_operators(overhead):
    return sorted((k for k, v in overhead.items() if v > LIMIT), key=lambda k: overhead[k], reverse=True)

def process_file(file_path, output_file):
    with open(file_path, 'r') as f:
        file_content = f.read()

    is_op_file = "_operators.txt" in file_path
    label = "SGX-Files" if is_op_file else "SGX"
    split_sections = file_content.split(label)

    systems = [
        parse_execution_times(section) if idx == 0 else (
            parse_execution_times_operators(section) if is_op_file else parse_execution_times(section)
        )
        for idx, section in enumerate(split_sections)
    ]

    s1 = {entry["operator"]: entry["total_time_ms"] for entry in systems[0]}
    s2 = {entry["operator"]: entry["total_time_ms"] for entry in systems[1]}

    if is_op_file:
        s1 = find_keys(s1)
        s2 = find_keys(s2)

    s1, s2 = update_operator_times(s1, s2, file_path[:-4]) 

    if not check_ops(s1, s2):
        return

    overhead = calculate_overhead(s1, s2)
    heaviest = heaviest_operators(overhead)

    with open(output_file, "a") as out:
        if heaviest:
            out.write(f"Heaviest operators for {os.path.basename(file_path).capitalize()[:-4]}:\n")
        for op in heaviest:
            if overhead[op] > LIMIT:
                out.write(f"{op}: {overhead[op]:.2f}x\n")
        out.write("\n")

def main():
    base_dir = 'memory_intensive_ops'
    output_file = os.path.join(base_dir, 'operator_overhead.txt')

    for file in os.listdir(base_dir):
        path = os.path.join(base_dir, file)
        if os.path.isfile(path):
            process_file(path, output_file)

if __name__ == "__main__":
    main()