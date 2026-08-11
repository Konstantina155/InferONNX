import io
import os
import sys
import onnx
import onnx_tool
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast
from matplotlib.patches import Patch
from collections import Counter
from matplotlib.ticker import PercentFormatter

def find_shape(modelpath):
    model = modelpath
    i = 0
    shape = [0, 0, 0, 0]
    for input in model.graph.input:
        name = input.name
        tensor_type = input.type.tensor_type
        if (tensor_type.HasField("shape")):
            for d in tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    shape[i] = d.dim_value
                    i += 1
            break
        else:
            print ("unknown rank", end="")
    return shape, name

def capture_profiling(func, *args, **kwargs):
    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()
    func(*args, **kwargs)    
    printed_output = sys.stdout.getvalue()
    sys.stdout = stdout_backup

    filtered_output = ""
    capture = False
    for line in printed_output.splitlines():
        if capture:
            filtered_output += line + "\n"
        if "Name" in line:
            capture = True
            filtered_output += line + "\n"

    return filtered_output

def profile_to_df(graph, shape, model_name, exclude_ops=None):
    m = onnx_tool.Model(graph)
    if any(x in model_name for x in ["gpt2", "smol-llama", "mistral", "qwen"]):
        if "gpt2" in model_name:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{models_dir}gpt2/test_data_set_0/tokenizer.json")
        elif "llama" in model_name:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{models_dir}smol-llama-220M-GQA/test_data_set_0/tokenizer.json")
        elif "mistral" in model_name:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{models_dir}mistral-300M/test_data_set_0/tokenizer.json")
        else:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{models_dir}qwen2.5-0.5B/test_data_set_0/tokenizer.json")
        prompt = "Hi, how are you today?"
        input_ids = tokenizer.encode(prompt, return_tensors='np')
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        position_ids = np.arange(input_ids.shape[1])[None, :]
        if "gpt2" in model_name:
            m.graph.shape_infer({'input_ids': input_ids, 'attention_mask': attention_mask})
        else:
            m.graph.shape_infer({'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids})
    else:
        m.graph.shape_infer({'data': np.zeros(shape)})
    
    m.graph.profile()
    printed_output = capture_profiling(m.graph.print_node_map, exclude_ops=exclude_ops)
    
    df = pd.read_csv(io.StringIO(printed_output), delimiter='\s{2,}', engine='python')
    df.drop(0, inplace=True)
    columns_to_drop = ['Forward_MACs', 'PPercent', 'FPercent', 'MPercent', 'Params', 'InShape', 'OutShape']
    df.drop(columns=columns_to_drop, inplace=True) # did not drop 'Type' column
    df.set_index('Name', inplace=True)
    df['Memory'] = df['Memory'].str.replace(',', '')
    return df

def get_label(name_list, group):
    if group == "small_ops":
        return f"Other operations ({len(name_list)}x)"
    labels = ["/".join(name.split("/")[-2:]) for name in name_list]

    counts = Counter(labels)
    unique_labels = []
    for lbl in dict.fromkeys(labels):
        cnt = counts[lbl]
        unique_labels.append(f"{lbl} ({cnt}x)")
    if len(unique_labels) > 1:
        a = unique_labels[0]
        b = unique_labels[1]
        left1, right1 = a.split('/', 1)
        left2, right2 = b.split('/', 1)

        if right1 != right2:
            raise ValueError("Right parts are not the same")

        unique_labels = [f"({left1}|{left2})/{right1}"]
    return "".join(unique_labels)

def run_command_with_output(cmd, cwd=None):
    output = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (info_time, err) = output.communicate()
    if err:
        print(f"Error executing command: {cmd}")
        print(f"Error details: {err.decode('utf-8')}")
        sys.exit(1)
    return info_time.decode('utf-8')

def get_execution_times(models): # head_matmul, gather, average of gate + up, down, c_fc/gemm, c_proj/gemm, average of rest ops
    operations = ["head_matmul", "gather", "gate_proj_matmul", "up_proj_matmul", "down_proj_matmul", "c_fc_gemm", "c_proj_gemm", "all"]
    exec_time = {}
    for model in models:
        times = []
        for op in operations:
            if ("gpt" in model and "proj_matmul" in op) or ("gpt" not in model and "gemm" in op):
                average_time = 0.0
            else:
                result = run_command_with_output(f"python3 {inferONNX_path}/scripts/benchmarks/calculate_exec_time_of_ops.py {inferONNX_path} {op} {model} 1")
                average_time = result.rsplit("Average execution time:", 1)[-1].lstrip().split("\n")[0]
            times.append(float(average_time))
        exec_time[model] = times
        print(f"Results for {model}: {exec_time[model]}")
    return exec_time

def plot_memory_ops(all_models_memory, models, K=10):
    fig, ax = plt.subplots(figsize=(26, 16))
    ax_time = ax.twinx()
    bar_width, spacing = 0.22, 0.05

    colors_memory = [
        '#F7D469',
        '#E08A66',
        '#8FC1E3',
        '#9E80A3',
    ]

    # Preprocess data for all models
    model_data = []
    model_names = list(MODEL_TO_NAMES.values())
    for model_name, memory_df in zip(model_names, all_models_memory.values()):
        df = memory_df.reset_index().copy()
        df["Group"] = df["Memory_MB"].apply(lambda x: "small_ops" if x < 7 else x)
        small_ops_df = df[df["Group"] == "small_ops"].copy()
        small_ops_df["Mean"] = small_ops_df["Memory_MB"]
        small_ops_df["Min"] = small_ops_df["Memory_MB"]
        small_ops_df["Max"] = small_ops_df["Memory_MB"]
        groups = df.groupby("Group").agg({"Name": list, "Memory_MB": "mean"}).reset_index()
        groups["Min"] = groups["Group"].apply(lambda g: small_ops_df["Memory_MB"].min() if g=="small_ops" else None)
        groups["Max"] = groups["Group"].apply(lambda g: small_ops_df["Memory_MB"].max() if g=="small_ops" else None)
        groups["Label"] = groups.apply(lambda row: get_label(row["Name"], row["Group"]), axis=1)
        groups = groups.sort_values(by="Memory_MB", ascending=False).head(K)
        model_data.append((model_name, groups))
    
    markers = ['o', 's', 'D', 'v']
    exec_time = get_execution_times(models)

    for i, (model_name, groups) in enumerate(model_data):
        x = np.arange(len(groups)) + i * bar_width
        ax.bar(x, groups["Memory_MB"], width=bar_width, color=colors_memory[i % len(colors_memory)], edgecolor='black')

        key = models[model_names.index(model_name)]
        exec_time_model = exec_time[key]

        for j, row in groups.iterrows():
            if row["Group"] == "small_ops":
                mean_val = row["Memory_MB"]
                lower_err = mean_val - row["Min"]
                upper_err = row["Max"] - mean_val
                ax.errorbar(x=x[j], y=mean_val, yerr=[[lower_err], [upper_err]], fmt="none", ecolor="red", capsize=13)

        if "gpt" in model_name:
            exec_vals = np.array([exec_time_model[0], exec_time_model[1], exec_time_model[5], exec_time_model[6], exec_time_model[7]])
        else:
            mean = (exec_time_model[2] + exec_time_model[3]) / 2
            exec_vals = np.array([exec_time_model[0], exec_time_model[1], mean, exec_time_model[4], exec_time_model[7]])
        exec_vals_sorted = np.sort(exec_vals)[::-1]
        cum_data = np.cumsum(exec_vals_sorted) / exec_vals.sum() * 100
        x_pareto = np.arange(len(exec_vals_sorted)) + i * bar_width
        ax_time.plot(x_pareto, cum_data, marker=markers[i % len(markers)], linestyle='-', linewidth=3, markersize=14, color="darkred")
    
    legend_handles = [
        Patch(facecolor=colors_memory[0], edgecolor='black', linewidth=2, label=model_names[0]),
        Patch(facecolor=colors_memory[1], edgecolor='black', linewidth=2, label=model_names[1]),
        Patch(facecolor=colors_memory[2], edgecolor='black', linewidth=2, label=model_names[2]),
        Patch(facecolor=colors_memory[3], edgecolor='black', linewidth=2, label=model_names[3]),
    ]
    ax.legend(handles=legend_handles, fontsize=31.3, bbox_to_anchor=(1.024, 1.15), ncol=len(model_names), frameon=False)

    ticks = []
    tick_labels = []
    for i, (_, groups) in enumerate(model_data):
        x = np.arange(len(groups)) + i * bar_width
        ticks.extend(x.tolist())
        tick_labels.extend(groups["Label"].tolist())

    ticks, tick_labels = zip(*sorted(zip(ticks, tick_labels)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=30)

    # Formatting
    ax_time.set_ylabel("Cumulative execution", fontsize=43, labelpad=30)
    ax_time.yaxis.set_major_formatter(PercentFormatter())
    ax_time.set_ylim(87, 100)
    ax_time.tick_params(axis='y', labelsize=35)

    ax.set_ylabel("Memory size (MB)", fontsize=43, labelpad=30)
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=35)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f"{inferONNX_path}/results/figure2_journal.pdf", format='pdf', dpi=600)

def find_op_types(memory_df):
    types_list = memory_df['Type'].tolist()
    print(set(types_list))
    print(len(set(types_list)))

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 onnx_op_memory_usage.py <path_to_inferONNX>")
        exit(1)

    global inferONNX_path, models_dir, MODEL_TO_NAMES
    inferONNX_path = sys.argv[1]

    models_dir = os.path.join(inferONNX_path, "models/")
    MODEL_TO_NAMES = {
        'gpt2': 'GPT-2 (0.124B)',
        'smol-llama-220M-GQA': 'LLaMA 2 (0.22B)',
        'mistral-300M': 'Mistral (0.3B)',
        'qwen2.5-0.5B': 'Qwen2.5 (0.5B)',
    }
    models = list(MODEL_TO_NAMES.keys())
    all_models_memory = {}

    for model in models:
        model_name = models_dir + model + '/' + model + ".onnx"
        modelpath = onnx.load(model_name)
        shape, _ = find_shape(modelpath)
        memory_df = profile_to_df(modelpath, shape, model_name)
        memory_df.to_csv(model + '_detailed.csv')

        memory_df = memory_df[
            (memory_df.index != "Total") &
            (memory_df["Type"] != "Constant")
        ].copy()
        memory_df["Memory_MB"] = memory_df["Memory"].astype(int) / (1024 * 1024)
        #sorted_memory_df = memory_df.sort_values(by="Memory_MB", ascending=False)

        #find_op_types(memory_df)
        all_models_memory[model] = memory_df

    plot_memory_ops(all_models_memory, models, K=100)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
