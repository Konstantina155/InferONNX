from collections import defaultdict
import sys
import os

def parse_execution_times_operators(file_content, is_nlp, file_path):
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

            if (is_nlp and len(parts) > 1 and "adhoc" in parts[1]):
                continue

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

    if is_nlp:
        result = [{"operator": op, "total_time_ms": time} for op, time in operators.items()]
        if debug:
            print(result)
            print(len(result))
        
        grouped_keys = defaultdict(list)
        for key in operators.keys():
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
        if debug:
            print(grouped_keys)
        
        k = 0
        for base_key, related_keys in grouped_keys.items():
            if len(related_keys) > 1:
                combined_value = sum(operators[key] for key in related_keys)
                operators[base_key] = combined_value
                for key in related_keys:
                    if key != base_key:
                        k += 1
                        operators.pop(key)
                
        keys = list(operators.keys())
        i = 0
        while i < len(keys):
            key = keys[i]

            if key in ["input_ids", "attention_mask", "position_ids"]:
                if i + 1 < len(keys):
                    next_key = keys[i + 1]
                    operators[next_key] += operators[key]
                operators.pop(key)
                keys.pop(i)
                continue
            i += 1
 
        if "smol-llama" in file_path:
            layers = 10
        elif "gpt2" in file_path:
            layers = 12
        else:
            layers = 24

        if debug:
            heaviest = sorted((k for k, v in operators.items() if v > 5), key=lambda k: operators[k], reverse=True)
            for op in heaviest:
                print(f"{op}: {operators[op]:.2f}x")
        heavy_ops = ["/lm_head/MatMul"]
        if "gpt" not in file_path:
            heavy_ops += ["/model/embed_tokens/Gather"]
            heavy_ops += [f"/model/layers.{i}/mlp/gate_proj/MatMul" for i in range(layers)]
            heavy_ops += [f"/model/layers.{i}/mlp/up_proj/MatMul" for i in range(layers)]
            heavy_ops += [f"/model/layers.{i}/mlp/down_proj/MatMul" for i in range(layers)]
        else:
            heavy_ops += ["/transformer/wte/Gather"]
        heavy_ops_tuple = tuple(heavy_ops)
        for op in list(operators.keys()):
            if op.startswith(heavy_ops_tuple):
                operators.pop(op)

    result = [{"operator": op, "total_time_ms": time} for op, time in operators.items()]
    if debug:
        print(result)
        print(len(result))

    times = list(operators.values())
    return sum(times) / len(times)

def parse_execution_times(file_content, operation, file_path):
    lines = file_content.splitlines()
    current_time = 0.0
    time = []

    lines = lines[2:]
    if operation == "head_matmul":
        if "gpt" in file_path:
            start_point = "/transformer/Reshape_3_output_0"
        else:
            start_point = "/model/norm/Mul_1_output_0"
        end_point = "logits_0"
    elif operation == "gate_proj_matmul":
        start_point = "/mlp/gate_proj/MatMul"
        end_point = ""
    elif operation == "up_proj_matmul":
        start_point = "mlp/up_proj/MatMul"
        end_point = ""
    elif operation == "down_proj_matmul":
        start_point = "/mlp/down_proj/MatMul"
        end_point = ""
    elif operation == "c_fc_gemm":
        start_point = "/mlp/c_fc/Gemm"
        end_point = ""
    elif operation == "c_proj_gemm":
        start_point = "/mlp/c_proj/Gemm"
        end_point = ""
    else:
        if "gpt" in file_path:
            start_point = "/transformer/Reshape_output_0"
            end_point = "/transformer/wte/Gather_0"
            lines = lines[:-1] if start_point in lines else lines[:2]
        else:
            start_point = '"/model/embed_tokens/Gather" Gather'
            end_point = "logits_0"
            lines = lines[:2]
    count = 0
    count_layers = 0
    if debug:
        print(lines)

    for line in lines:
        if "Running step" in line:
            parts = line.split('"')

            if "adhoc" in parts[1] and "matmul" not in operation and "gather" not in operation: break

            if (parts[1] == start_point) or (parts[1] == end_point and parts[2] == " Source") or (count == 2 and start_point not in parts[1]):
                if current_time != 0.0:
                    time.append(current_time)
                    count_layers += 1
                current_time = 0.0

            if start_point in parts[1] and (parts[2] == " LirMatMulUnary" or \
                (parts[2] == " MatMatMulPack")): count = 1

        elif "takes" in line and count <= 1:
            if count == 1:
                count = 2
            time_taken = float(line.split("takes")[1].split("ms")[0].strip())
            current_time += time_taken
        else:
            continue

    if "proj_matmul" not in operation and "gemm" not in operation:
        time.append(current_time)
        result = sum(time)
    else:
        result = sum(time) / len(time)
    if debug:
        print(time, len(time))
    return result

def find_text_above_gemm(extracted_content, text):
    top_half = extracted_content.split(text)[0]
    return top_half.split('\n')[-3:-2]

def process_file(file_path, occurrence, operation):
    with open(file_path, 'r') as f:
        file_content = f.read()

    sgx_indices = [i for i in range(len(file_content)) if file_content.startswith('SGX', i)]

    if len(sgx_indices) < occurrence:
        return "Not enough 'SGX' occurrences in the file."

    start = sgx_indices[occurrence - 1]
    end = sgx_indices[occurrence] if occurrence < len(sgx_indices) else len(file_content)
    extracted_content = file_content[start:end]

    if operation == "head_matmul":
        if "gpt" not in file_path:
            split_point = '"/model/norm/Mul_1" Mul'
        else:
            split_point = '"/transformer/ln_f/Add_1" Add'
        parts = extracted_content.split(split_point, 1)
    elif operation == "gate_proj_matmul":
        split_point = '"/model/layers.0/post_attention_layernorm/Mul_1" Mul'
        parts = extracted_content.split(split_point, 1)
    elif operation == "up_proj_matmul":
        split_point = '"/model/layers.0/mlp/act_fn/Mul" MulUnicast'
        parts = extracted_content.rsplit(split_point, 1)
    elif operation == "down_proj_matmul":
        split_point = '"/model/layers.0/mlp/Mul" MulUnicast'
        parts = extracted_content.split(split_point, 1)
    elif operation == "c_fc_gemm":
        top_half = extracted_content.split('"/transformer/h.0/mlp/c_fc/Gemm')[0]
        split_point = top_half.split('\n')[-3:-2][0]
        parts = extracted_content.split(split_point, 1)
    elif operation == "c_proj_gemm":
        top_half = extracted_content.split('"/transformer/h.0/mlp/c_proj/Gemm')[0]
        split_point = top_half.split('\n')[-3:-2][0]
        parts = extracted_content.split(split_point, 1)
    elif operation == "gather":
        split_point = 'input_ids'
        if "gpt" not in file_path or occurrence == 1:
            parts = extracted_content.split(split_point, 1)
        else:
            split_point_end = '"/transformer/wte/Gather_output_0" Source'
            parts_at_end = extracted_content.split(split_point_end, 1)
            parts = parts_at_end[0].split(split_point, 1)
    else:
        down_half = extracted_content.rsplit('"adhoc', 1)
        lines_below = down_half[1].split('\n', 3)
        if "Slice" in lines_below[0] or "takes" in lines_below[0]:
            split_point = lines_below[1]
        else:
            split_point = lines_below[2]
        idx = extracted_content.find(split_point)
        parts = [extracted_content[:idx], extracted_content[idx:]]
    
    content_below_target = parts[1]
    if len(parts) < 2:
        return "Target line not found in the extracted content."

    is_nlp = "gpt" in file_path or "llama" in file_path or "mistral" in file_path or "qwen" in file_path
    if operation == "all":
        exec_time = parse_execution_times_operators(content_below_target, is_nlp, file_path)
    else:
        exec_time = parse_execution_times(content_below_target, operation, file_path)
    print(f"Average execution time: {exec_time}")

def main():
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("python3 calculate_exec_time_of_ops.py <inferONNX_path> <gather/head_matmul/gate_proj_matmul/up_proj_matmul/down_proj_matmul/c_fc_gemm/c_proj_gemm/all> <model_name> <occurence_from_beginning> <debug/''>")
        sys.exit(1)

    global debug
    inferONNX_path = sys.argv[1]
    operation = sys.argv[2]
    file_name = sys.argv[3]
    occurence = int(sys.argv[4])
    assert file_name in ["gpt2", "smol-llama-220M-GQA", "mistral-300M", "qwen2.5-0.5B"], "Model name is not correct"
    assert operation in ["head_matmul", "gather", "up_proj_matmul", "gate_proj_matmul", "down_proj_matmul", "c_fc_gemm", "c_proj_gemm", "all"], "Operation should be gather or head_matmul or {gate/up/down}_proj/matmul / {c_fc/c_proj}/Gemm or all"
    assert occurence > 0, "Occurence is not > 0"
    
    if len(sys.argv) == 6:
        assert sys.argv[5] == "debug", "Debug mode is not correct"
        debug = True
    else:
        debug = False

    file_path = f'{inferONNX_path}/memory_intensive_ops/{file_name}.txt'

    if os.path.isfile(file_path):
        print("Processing file:", file_path, "for operation:", operation)
        process_file(file_path, occurence, operation)

if __name__ == "__main__":
    main()

