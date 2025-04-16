import re
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from decimal import Decimal, ROUND_HALF_UP

if len(sys.argv) != 4:
    print("Usage: python3 create_plots.py <inferONNX_path> <partitions_folder> <number_of_runs>")
    exit(1)

inferONNX_path = sys.argv[1]
partitions_folder = sys.argv[2]
number_of_runs = sys.argv[3]

if not os.path.exists(f"{inferONNX_path}/results"):
    os.mkdir(f"{inferONNX_path}/results")

models = ["squeezenet1.0-7.onnx", "squeezenet1.0-7_split0.onnx", "mobilenetv2-7.onnx", "mobilenetv2-7_split0.onnx", 
          "densenet-7.onnx", "densenet-7_split0.onnx", "efficientnet-lite4-11.onnx", "efficientnet-lite4-11_split0.onnx",
          "inception-v3-12.onnx", "inception-v3-12_split0.onnx", "resnet101-v2-7.onnx", "resnet101-v2-7_split0.onnx",
          "resnet152-v2-7.onnx", "resnet152-v2-7_split0.onnx", "efficientnet-v2-l-18.onnx", "efficientnet-v2-l-18_split0.onnx"]
model_names = ['SqueezeNet', 'MobileNet\nV2', 'DenseNet\n121', 'EfficientNet\nLite4', 'Inception\nV3', 'ResNet101\nV2',
               'ResNet152\nV2', 'EfficientNet\nV2']

num_partitions = []
for model in models:
    if "split" in model: continue
    result = subprocess.run(f'ls {inferONNX_path}/models/{model[:-5]}/{partitions_folder} | wc -w', shell=True, text=True, capture_output=True)
    num_partitions.append(int(result.stdout.strip()))

files = [f"{inferONNX_path}/src/server_without_tls/inference_time_cpu_on_disk_no_aes.txt",
         f"{inferONNX_path}/src/server_without_tls/inference_time_cpu_memory_only_no_aes.txt",
         f"{inferONNX_path}/src/server_with_tls/inference_time_outside_occlum_on_disk_no_aes.txt",
         f"{inferONNX_path}/src/server_with_tls/inference_time_in_occlum_memory_only_aes.txt",
         f"{inferONNX_path}/src/server_with_tls/inference_time_in_occlum_on_disk_aes.txt"]
configurations = ['CPU-on Disk', 'CPU-on Memory', 'CPU-TLS/SSL enabled', 'SGX-on Memory', 'SGX-on Disk']

def change_configuration_name(configuration):
    if " " in configuration:
        return configuration.replace(" ", "-")
    return configuration
    
def change_model_name(path):
    if 'split' in path:
        if 'part' in path:
            base_name = path.replace('_part0.onnx', '')
        else:
            base_name = path.replace('.onnx', '')
    else:
        base_name = path.replace('.onnx', '')
    return base_name
    

def parse_time(configuration, model_name, time, df, i):
    output_str = time.decode('utf-8')

    if i != -1:
        numbers = re.findall(r'(\d+\.\d{1}|\d+)\s*ms', output_str)
        numbers = [int(num) for num in numbers]
        if len(numbers) == 3:
            df.loc[i] = [configuration, change_model_name(model_name), numbers[0], numbers[1], numbers[2]]
            return None
    else:
        numbers = re.findall(r'(\d*\.?\d+)\s*ms', output_str)
        numbers = [float(num) for num in numbers]
        if len(numbers) > 5:
            return numbers[5:]

def plot_whole_vs_partitions(color_whole_model, color_split_model, configuration, size_models):
    df = df_total[df_total['Configuration'] == configuration]
    if size_models == "small":
        models = ['squeezenet1.0-7',  'mobilenetv2-7', 'densenet-7', 'efficientnet-lite4-11']
        model_names_str = ['SqueezeNet\n1.0', 'MobileNet\nV2', 'DenseNet\n121', 'EfficientNet\nLite4']
    elif size_models == "large":
        models = ['inception-v3-12', 'resnet101-v2-7', 'resnet152-v2-7', 'efficientnet-v2-l-18']
        model_names_str = ['Inception\nV3', 'ResNet101\nV2', 'ResNet152\nV2', 'EfficientNet\nV2']
    
    fig, ax = plt.subplots(figsize=(15, 12))
    bar_width = 0.3
    spacing = 0.1
    base_positions = np.arange(len(models)) * (bar_width * 2 + spacing)

    for i, model in enumerate(models):
        if model in df['Model'].values and model + '_split0' in df['Model'].values:
            data = df[df['Model'] == model]
            data_split = df[df['Model'] == model + '_split0']
            ax.bar(base_positions[i] - bar_width / 2, data['Total Client Time'].values[0], bar_width, color=color_whole_model, linewidth=2)
            ax.bar(base_positions[i] + bar_width / 2, data_split['Total Client Time'].values[0], bar_width, color=color_split_model, linewidth=2)
    
            ax.text(base_positions[i] - bar_width / 2, data['Total Client Time'].values[0] + 10, str(round(data['Total Client Time'].values[0], 2)), ha='center', va='bottom', fontsize=26)
            ax.text(base_positions[i] + bar_width / 2, data_split['Total Client Time'].values[0] + 10, str(round(data_split['Total Client Time'].values[0], 2)), ha='center', va='bottom', fontsize=26)

    ax.set_ylabel('Inference Time (ms)', fontsize=36, labelpad=30)

    legend_handles = [
        Patch(facecolor=color_whole_model, edgecolor='black', linewidth=2, label=f'InferONNX-on Disk with SGX - Whole model'),
        Patch(facecolor=color_split_model, edgecolor='black', linewidth=2, label=f'InferONNX-on Disk with SGX - Partitions of model'),
    ]
    ax.legend(handles=legend_handles, fontsize=28, loc='upper center', bbox_to_anchor=(0.4, 1.17), ncol=len(legend_handles)/2, frameon=False)


    ax.set_xticks(base_positions)
    ax.set_xticklabels(model_names_str, fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    plt.xticks()
    
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(f"{inferONNX_path}/results/figure5_{size_models}_models.pdf", format='pdf')
    plt.show()

def plot_sgx_disk_vs_sgx_memory_vs_cpu_paper(size_models):
    df_sgx = df_total[df_total['Configuration'] == 'SGX-on Disk']
    seen = OrderedDict()
    for model in df_sgx['Model']:
        base_model = model.split('_split0')[0]
        seen[base_model] = None
    models = list(seen.keys())
    if size_models == "small":
        models = ['squeezenet1.0-7',  'mobilenetv2-7', 'densenet-7', 'efficientnet-lite4-11']
        model_names_str = ['SqueezeNet\n1.0', 'MobileNet\nV2', 'DenseNet\n121', 'EfficientNet\nLite4']
    else: 
        models = ['inception-v3-12', 'resnet101-v2-7', 'resnet152-v2-7', 'efficientnet-v2-l-18']
        model_names_str = ['Inception\nV3', 'ResNet101\nV2', 'ResNet152\nV2', 'EfficientNet\nV2']

    fig, ax = plt.subplots(figsize=(15, 12))
    bar_width = 0.3
    spacing = 0.1
    base_positions = np.arange(len(models)) * (bar_width * 3 + spacing)

    colors = [
        '#B455FF',
        '#A237FF',
        '#9019FF'
    ]


    current_configs = ['SGX-on Disk', 'SGX-Memory only', 'CPU']

    for i, model in enumerate(models):
        df_sgx_disk = df_total[df_total['Configuration'] == 'SGX-on Disk']
        df_sgx_caching = df_total[df_total['Configuration'] == 'SGX-on Memory']
        df_cpu_ssl = df_total[df_total['Configuration'] == 'CPU-TLS/SSL enabled']

        if model in df_sgx['Model'].values and model in df_sgx_caching['Model'].values and model in df_cpu_ssl['Model'].values:
            data_sgx_disk = df_sgx_disk[df_sgx_disk['Model'] == model]
            data_sgx_caching = df_sgx_caching[df_sgx_caching['Model'] == model]
            data_cpu_ssl = df_cpu_ssl[df_cpu_ssl['Model'] == model]
            ax.bar(base_positions[i] + 0 * bar_width, data_sgx_disk['Total Client Time'].values[0], bar_width, color=colors[0])
            ax.bar(base_positions[i] + 1 * bar_width, data_sgx_caching['Total Client Time'].values[0], bar_width, color=colors[1])
            ax.bar(base_positions[i] + 2 * bar_width, data_cpu_ssl['Total Client Time'].values[0], bar_width, color=colors[2])
            
            ax.text(base_positions[i] + 0 * bar_width, data_sgx_disk['Total Client Time'].values[0] + 10, str(round(data_sgx_disk['Total Client Time'].values[0], 2)), ha='center', va='bottom', fontsize=24)
            ax.text(base_positions[i] + 1 * bar_width, data_sgx_caching['Total Client Time'].values[0] + 10, str(round(data_sgx_caching['Total Client Time'].values[0], 2)), ha='center', va='bottom', fontsize=24)
            ax.text(base_positions[i] + 2 * bar_width, data_cpu_ssl['Total Client Time'].values[0] + 10, str(round(data_cpu_ssl['Total Client Time'].values[0], 2)), ha='center', va='bottom', fontsize=24)
            
    ax.set_ylabel('Inference Time (ms)', fontsize=36, labelpad=30)

    legend_handles = [
        Patch(facecolor=colors[0], edgecolor='black', linewidth=2, label=f'InferONNX-on Disk with SGX'),
        Patch(facecolor=colors[1], edgecolor='black', linewidth=2, label=f'InferONNX-Memory only with SGX'),
        Patch(facecolor=colors[2], edgecolor='black', linewidth=2, label=f'InferONNX-without SGX'),
    ]
    ax.legend(handles=legend_handles, fontsize=28, loc='upper center', bbox_to_anchor=(0.4, 1.2), ncol=2, frameon=False)

    ax.set_xticks(base_positions + (len(current_configs) - 1) * bar_width / 2)
    ax.set_xticklabels(model_names_str, fontsize=30)
    if size_models == "small":
        plt.ylim(0, 3000)

    ax.tick_params(axis='y', labelsize=30)
    plt.xticks()

    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(f"{inferONNX_path}/results/figure4_{size_models}_models.pdf", format='pdf')
    plt.show()

def table_cpu_memory_vs_cpu_disk():
    rows = []

    model_names = {
        'squeezenet1.0-7': 'SqueezeNet 1.0',
        'mobilenetv2-7': 'MobileNet V2',
        'densenet-7': 'DenseNet121',
        'efficientnet-lite4-11': 'EfficientNet Lite4',
        'inception-v3-12': 'Inception V3',
        'resnet101-v2-7': 'ResNet101 V2',
        'resnet152-v2-7': 'ResNet152 V2',
        'efficientnet-v2-l-18': 'EfficientNet V2'
    }

    for model in models:
        model = model[:-5]
        if 'split' in model: continue
        df_cpu_memory = df_total[df_total['Configuration'] == 'CPU-on Memory']
        df_cpu_disk = df_total[df_total['Configuration'] == 'CPU-on Disk']

        if (model in df_cpu_memory['Model'].values and model in df_cpu_disk['Model'].values):
            data_cpu_memory = df_cpu_memory[df_cpu_memory['Model'] == model]
            data_cpu_disk = df_cpu_disk[df_cpu_disk['Model'] == model]

            rows.append({
                'Model': model_names[model],
                'InferONNX-Memory only w/o SGX (ms)': str(round(data_cpu_memory['Total Client Time'].values[0], 2)),
                'InferONNX-on Disk w/o SGX (ms)': str(round(data_cpu_disk['Total Client Time'].values[0], 2)),
            })

    df_result = pd.DataFrame(rows)
    df_result.to_csv(f'{inferONNX_path}/results/table2.csv', index=False)

def plot_inference_time_breakdown():
    df_sgx_disk = df_total[df_total['Configuration'] == 'SGX-on Disk']
    inference = []
    entire_inference = []
    selected_models = ["mobilenetv2-7.onnx", "densenet-7.onnx", "inception-v3-12.onnx", "resnet152-v2-7.onnx", "efficientnet-v2-l-18.onnx"]
    names = ['MobileNet\nV2', 'DenseNet121', 'Inception\nV3', 'ResNet152\nV2', 'EfficientNet\nV2'] 
    for selected_model in selected_models:
        data_sgx_disk = df_sgx_disk[df_sgx_disk['Model'] == selected_model[:-5]]
        inference.append(data_sgx_disk['Run Model Time'].values[0])
        entire_inference.append(data_sgx_disk['Inference Time'].values[0])
    loading = np.array(entire_inference) - np.array(inference)


    index_models = np.arange(len(selected_models))
    width = 0.45
    fig, ax = plt.subplots(figsize =(8, 4))
    p1 = plt.bar(index_models, loading, width, color ='#3594CC', edgecolor ='black')
    p2 = plt.bar(index_models, inference, width, color='#D31F11', edgecolor ='black', bottom = loading)

    plt.subplots_adjust(right=0.95, top=0.9, left=0.14, bottom=0.15)
    plt.ylabel('Execution time (ms)', fontsize=16)
    plt.xticks(index_models, names, fontsize=14)
    plt.yticks(fontsize=14)  
    plt.legend((p1[0], p2[0]), ('Load and decrypt model', 'Inference process'), fontsize=17)

    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    plt.tight_layout()

    if not os.path.exists('results/'):
        os.mkdir('results/')
    plt.savefig('results/figure1.pdf', format='pdf')

if __name__ == '__main__':
    df_total = pd.DataFrame(columns=['Configuration', 'Model', 'Run Model Time', 'Inference Time', 'Total Client Time'])
    index_df = 0
    index_model_split = -1
    found = 0
    for model in models:
        index_config = 0
        if 'split' in model:
            index_model_split += 1
            number_partitions = num_partitions[index_model_split]
        else:
            number_partitions = 1
        for file in files:
            if 'split' in model:
                if "inference_time_in_occlum_on_disk_aes.txt" in file:
                    command = f"python3 {inferONNX_path}/scripts/calculate_avg_time.py " + file + " " + model + " " + str(number_partitions) + " " + str(number_of_runs) + " 1"
                else:
                    index_config += 1
                    index_df += 1
                    continue
            else:
                occurence = 1
                command = f"python3 {inferONNX_path}/scripts/calculate_avg_time.py " + file + " " + model + " 1 " + str(number_of_runs) + " " + str(occurence)
            output = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
            (time, err) = output.communicate()

            configuration = configurations[index_config]
            parse_time(configuration, model, time, df_total, index_df)
            index_config += 1
            index_df += 1

    plot_whole_vs_partitions('#54a1a1', '#1f6f6f', 'SGX-on Disk', 'small') #blue
    plot_whole_vs_partitions('#54a1a1', '#1f6f6f', 'SGX-on Disk', 'large') #blue

    plot_sgx_disk_vs_sgx_memory_vs_cpu_paper('small') #purple
    plot_sgx_disk_vs_sgx_memory_vs_cpu_paper('large') #purple

    table_cpu_memory_vs_cpu_disk()

    plot_inference_time_breakdown()