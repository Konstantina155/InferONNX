import re
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

def parse_time(configuration, model_name, inference_time, df, index):
    output_str = inference_time.decode('utf-8')
    metric = 's'

    numbers = re.findall(rf'(\d+\.\d{1}|\d+)\s*{metric}', output_str)
    numbers = [int(num) for num in numbers]

    df.loc[index] = [configuration, model_name.replace('.onnx', ''), num_tokens, *numbers]

def plot_whole_vs_partitions(df_total, text=False):
    models, labels = ['gpt2', 'smol-llama-220M-GQA', 'mistral-300M', 'qwen2.5-0.5B'], ['GPT-2 (0.124B)', 'LLaMA 2 (0.22B)', 'Mistral (0.3B)', 'Qwen2.5 (0.5B)']
    fig, ax = plt.subplots(figsize=(16, 12))
    bar_width, spacing = 0.3, 0.1

    colors = [
        '#d4876a',
        '#a63c1f', 
        '#6e2010'
    ]

    num_configurations = 3
    df = df_total[
        (df_total['Configuration'] == 'SGX-on Disk') &
        (df_total['Number of tokens generated'] == num_tokens)
    ]
    positions = np.arange(len(models)) * (bar_width * num_configurations + spacing)
   
   
    for i, model in enumerate(models):
        base = df[df['Model'] == model]
        split = df[df['Model'] == model + '_split0']
        if not base.empty and not split.empty:
            first_split = split.iloc[0]
            second_split = split.iloc[1]
            ax.bar(positions[i] + 0 * bar_width, base['Total Client Time'].values[0], bar_width, color=colors[0], edgecolor='black')
            ax.bar(positions[i] + 1 * bar_width, second_split['Total Client Time'], bar_width, color=colors[1], edgecolor='black')
            ax.bar(positions[i] + 2 * bar_width, first_split['Total Client Time'], bar_width, color=colors[2], edgecolor='black')

            if text:
                first_split = split.iloc[0]
                second_split = split.iloc[1]
                ax.text(positions[i] + 0 * bar_width, base['Total Client Time'].values[0] + 10, str(round(base['Total Client Time'].values[0], 2)), ha='center', va='bottom', fontsize=24)
                ax.text(positions[i] + 1 * bar_width, second_split['Total Client Time'] + 10, str(round(second_split['Total Client Time'], 2)), ha='center', va='bottom', fontsize=24)
                ax.text(positions[i] + 2 * bar_width, first_split['Total Client Time'] + 10, str(round(first_split['Total Client Time'], 2)), ha='center', va='bottom', fontsize=24)

    ax.set_ylabel('Time to First Token (s)', fontsize=43, labelpad=30)
    legend_handles = [
        Patch(facecolor=colors[0], edgecolor='black', linewidth=2, label=f'Full model execution'),
        Patch(facecolor=colors[1], edgecolor='black', linewidth=2, label=f'Inter-op partitioning'),
        Patch(facecolor=colors[2], edgecolor='black', linewidth=2, label=f'Inter- & intra-op partitioning'),
    ]
    ax.legend(handles=legend_handles, title='ShardLLM', fontsize=31.5, title_fontsize=31.5, loc='upper left', ncol=1, edgecolor='grey', framealpha=1, handleheight=1.2) # 30.7, 30

    ax.set_xticks(np.arange(len(models)) * (0.3 * num_configurations + 0.1) + 2 * 0.3 / 2)
    ax.set_xticklabels(labels, fontsize=35)
    plt.tick_params(axis='y', labelsize=40)
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(f"{inferONNX_path}/results/figure4_journal.pdf", format='pdf')
    plt.show()

def table_sgx_disk_vs_sgx_memory_vs_cpu_memory(df_total):
    rows = []

    model_names = {
        'gpt2': 'GPT-2 (0.124B)',
        'smol-llama-220M-GQA': 'LLaMA 2 (0.22B)',
        'mistral-300M': 'Mistral (0.3B)',
        'qwen2.5-0.5B': 'Qwen2.5 (0.5B)'
    }

    for model in models:
        model = model[:-5]
        if 'split' in model: continue
        df_sgx_disk = df_total[
            (df_total['Configuration'] == 'SGX-on Disk') &
            (df_total['Number of tokens generated'] == num_tokens)
        ]
        df_sgx_on_memory = df_total[
            (df_total['Configuration'] == 'SGX-on Memory') &
            (df_total['Number of tokens generated'] == num_tokens)
        ]
        df_cpu_ssl = df_total[
            (df_total['Configuration'] == 'CPU-TLS/SSL enabled') &
            (df_total['Number of tokens generated'] == num_tokens)
        ]

        if (model in df_sgx_disk['Model'].values and model in df_sgx_on_memory['Model'].values and model in df_cpu_ssl['Model'].values):
            data_sgx_disk = df_sgx_disk[df_sgx_disk['Model'] == model]
            data_sgx_memory = df_sgx_on_memory[df_sgx_on_memory['Model'] == model]
            data_cpu_memory = df_cpu_ssl[df_cpu_ssl['Model'] == model]

            rows.append({
                'Model': model_names[model],
                'TTFT in Intel SGX (Loaded from disk)': str(round(data_sgx_disk['Total Client Time'].values[0], 2)),
                'TTFT in Intel SGX (Preloaded in memory)': str(round(data_sgx_memory['Total Client Time'].values[0], 2)),
                'TTFT on CPU': str(round(data_cpu_memory['Total Client Time'].values[0], 2)),
            })

    df_result = pd.DataFrame(rows)
    df_result.to_csv(f'{inferONNX_path}/results/table1_journal.csv', index=False)
    
def plot_guardrail_boxplot(df_total):
    guardrail_order = ['Regex', 'DistilBert', 'Bert-base', 'Bert-large']

    colors = [
        '#3F6FB5',  # Blue — GPT-2
        '#C94C4C',  # Red — LLaMA 2
        '#D9A514',  # Gold — Mistral
        '#3D9B91'   # Teal — Qwen2.5
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.boxplot(
        data=df_total, x='Guardrail', y='Total Client Time (s)', hue='Model',
        order=guardrail_order, ax=ax, palette=colors
    )
    ax.set_xlabel('Guardrails', fontsize=30)
    ax.set_yscale('log')
    ax.set_ylabel('Time to First Token (s)', fontsize=30, labelpad=20)
    plt.tick_params(axis='y', labelsize=17)
    plt.tick_params(axis='x', labelsize=17)

    legend_handles = [
        Patch(facecolor=colors[0], edgecolor='black', linewidth=2, label='GPT-2 (0.124B)'),
        Patch(facecolor=colors[1], edgecolor='black', linewidth=2, label='LLaMA 2 (0.22B)'),
        Patch(facecolor=colors[2], edgecolor='black', linewidth=2, label='Mistral (0.3B)'),
        Patch(facecolor=colors[3], edgecolor='black', linewidth=2, label='Qwen2.5 (0.5B)'),
    ]
    ax.legend(handles=legend_handles, fontsize=17, loc='lower center', bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"{inferONNX_path}/results/guardrails_boxplot.pdf", format='pdf')
    plt.show()

def plot_distribution_dataset(df_total):
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.histplot(
        data=df_total,
        x='Length of prompt chars',
        binwidth=50,
        ax=ax
    )

    ax.set_xlabel('Prompt length', fontsize=30)
    ax.set_ylabel('Frequency', fontsize=30, labelpad=20)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, None)
    plt.tick_params(axis='y', labelsize=17)
    plt.tick_params(axis='x', labelsize=17)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"{inferONNX_path}/results/guardrails_prompt_length.pdf", format='pdf')
    plt.show()

def create_df():
    files_ = files
    configurations_ = configurations
    
    df_total = pd.DataFrame(columns=['Configuration', 'Model', 'Number of tokens generated', 'Run Model Time', 'Inference Time', 'Total Client Time'])
    index_df = 0
    index_model_split = -1
    previous_model = ""
    occurence_whole = 1
    for model in models:
        if 'split' in model:
            index_model_split += 1
            number_partitions = num_partitions[index_model_split]
        else:
            number_partitions = 1

        index_config = 0
        for file in files_:
            if not os.path.exists(file):
                index_config += 1
                index_df += 1
                continue

            if 'split' in model:
                if "occlum_on_disk_aes" in file:
                    occurence = 2 if previous_model == model else 1
                    command = f"python3 {inferONNX_path}/scripts/calculate_avg_time.py " + file + " " + model + " " + str(num_tokens) + " " + str(number_partitions) + " " + str(number_of_runs) + " " + str(occurence)
                    previous_model = model
                else:
                    index_config += 1
                    index_df += 1
                    continue
            else:
                if "occlum_memory_only" in file:
                    command = f"python3 {inferONNX_path}/scripts/calculate_avg_time.py " + file + " " + model + " " + str(num_tokens) + " 1 " + str(number_of_runs) + " " + str(occurence_whole)
                    occurence_whole += 1
                    if occurence_whole == 6: occurence_whole = 1
                else:
                    command = f"python3 {inferONNX_path}/scripts/calculate_avg_time.py " + file + " " + model + " " + str(num_tokens) + " 1 " + str(number_of_runs) + " 1"

            output = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            (inference_time, err) = output.communicate()
            if err:
                print(f"Error executing command: {command}")
                sys.exit(1)

            if int(number_of_runs) < 6:
                configuration = configurations_[index_config]
                parse_time(configuration, model, inference_time, df_total, index_df)
                index_config += 1
                index_df += 1
    if int(number_of_runs) > 6:
        df_total = pd.read_csv(f"{inferONNX_path}/results/guardrails_info.csv")
        df_total['Model'] = df_total['Model'].replace({
            'gpt2.onnx': model_names[0], 'smol-llama-220M-GQA.onnx': model_names[1], 'mistral-300M.onnx': model_names[2], 'qwen2.5-0.5B.onnx': model_names[3]
        })
        df_total['Guardrail'] = df_total['Guardrail'].replace({
            'baseline': 'Baseline', 'regex': 'Regex', 'distilbert': 'DistilBert',
            'bert-base': 'Bert-base', 'bert-large': 'Bert-large'
        })
    return df_total

def create_plots(df_total):
    print(df_total)
    plot_guardrail_boxplot(df_total)
    plot_distribution_dataset(df_total)

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 create_plots.py <inferONNX_path> <inter_partitions_folder> <intra_partitions_folder> <number_of_runs>")
        exit(1)

    global inferONNX_path, intra_partitions_folder, inter_partitions_folder, number_of_runs
    inferONNX_path = sys.argv[1]
    inter_partitions_folder = sys.argv[2]
    intra_partitions_folder = sys.argv[3]
    number_of_runs = sys.argv[4]

    os.makedirs(f"{inferONNX_path}/results", exist_ok=True)

    global models, model_names, files, configurations, num_partitions, num_tokens
    models = ["gpt2.onnx", "gpt2.onnx", "gpt2.onnx", "gpt2.onnx", "gpt2.onnx",
              "smol-llama-220M-GQA.onnx", "smol-llama-220M-GQA.onnx", "smol-llama-220M-GQA.onnx", "smol-llama-220M-GQA.onnx", "smol-llama-220M-GQA.onnx",
              "mistral-300M.onnx", "mistral-300M.onnx", "mistral-300M.onnx", "mistral-300M.onnx", "mistral-300M.onnx", 
              "qwen2.5-0.5B.onnx", "qwen2.5-0.5B.onnx", "qwen2.5-0.5B.onnx", "qwen2.5-0.5B.onnx", "qwen2.5-0.5B.onnx"]
    model_names = ['GPT-2 (0.124B)', 'LLaMA 2 (0.22B)', 'Mistral (0.3B)', 'Qwen2.5 (0.5B)']
    files = [f"{inferONNX_path}/src/server_with_tls/inference_time_in_occlum_memory_only_aes.txt"]
    configurations = ['SGX-on Memory']

    num_tokens = 1
    num_partitions = []
    for model in models:
        if "split" in model or not os.path.exists(f'{inferONNX_path}/models/{model[:-5]}/{intra_partitions_folder}'):
            continue

        if "gpt" in model or "llama" in model or "mistral" in model or "qwen" in model:
            result = subprocess.run(f'ls {inferONNX_path}/models/{model[:-5]}/{inter_partitions_folder} | wc -w', shell=True, text=True, capture_output=True)
            num_partitions.append(int(result.stdout.strip()))

        result = subprocess.run(f'ls {inferONNX_path}/models/{model[:-5]}/{intra_partitions_folder} | wc -w', shell=True, text=True, capture_output=True)
        num_partitions.append(int(result.stdout.strip()))

    print("Number of partitions for LLMs: ", num_partitions)
    df_total = create_df()

    create_plots(df_total)

if __name__ == '__main__':
    main()