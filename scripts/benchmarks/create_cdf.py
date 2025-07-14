import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

MODEL_NAMES = {
    'squeezenet1.0-7': 'SqueezeNet1.0',
    'mobilenetv2-7': 'MobileNet V2',
    'densenet-7': 'DenseNet121',
    'efficientnet-lite4-11': 'EfficientNet Lite4',
    'inception-v3-12': 'Inception V3',
    'resnet101-v2-7': 'ResNet101 V2',
    'resnet152-v2-7': 'ResNet152 V2',
    'efficientnet-v2-l-18': 'EfficientNet V2'
}

PARTITION_NAMES = {f'{k}_partitions': v for k, v in MODEL_NAMES.items()}

def parse_model_requirements(text):
    model_data = {}
    
    for block in text.split('Model:')[1:]:
        match = re.match(r'(\S+)\.txt', block.strip())
        if not match:
            continue

        model_key = match.group(1)
        if model_key in MODEL_NAMES:
            model_label = MODEL_NAMES[model_key]
        elif model_key in PARTITION_NAMES:
            model_label = PARTITION_NAMES[model_key]
        else:
            continue

        model_data[model_label] = {}
        
        percentage_lines = re.findall(r'exceeds ([\d.]+)MB by: ([\d.]+)%', block)
        
        for size, percentage in percentage_lines:
            if model_key in MODEL_NAMES:
                model_data[MODEL_NAMES[model_key]][size] = float(percentage)
            elif model_key in PARTITION_NAMES:
                model_data[PARTITION_NAMES[model_key]][size] = float(percentage)
    
    return model_data

def run_data_extraction(filename):
    extraction_script = ['python3', 'scripts/benchmarks/extract_ms_print_info.py']
    if 'partitions' in filename:
        extraction_script.append(sys.argv[1])
    subprocess.run(extraction_script, check=True)

def plot_cdf(ax, df, model_data, specific_xticks, specific_xticks_show):
    for model in df.index:
        model_data = df.loc[model]
        non_zero_data = model_data[model_data > 0]

        if not non_zero_data.empty:
            cdf = np.cumsum(non_zero_data) / non_zero_data.sum()
            ax.plot(non_zero_data.index, cdf, label=model, linewidth=2.5)

    valid_xticks = [x for x in specific_xticks_show if x in df.columns.astype(int)]
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xticks(valid_xticks)
    ax.set_xticklabels(valid_xticks, fontsize=40)
    ax.set_xticks(specific_xticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=6, width=1, colors='black')

def plot_model_data(df, model_data, specific_xticks, specific_xticks_show, is_partition=False):
    file_to_be_saved = 'results/figure3b' if is_partition else 'results/figure3a'

    fig, ax = plt.subplots(figsize=(16, 11))
    plot_cdf(ax, df, model_data, specific_xticks, specific_xticks_show)

    if is_partition:
        ax.legend(fontsize=39, loc='lower right', ncol=1)
    else:
        ax.set_ylabel('% of Total Execution Time', fontsize=43, labelpad=30)

    ax.set_xlabel('Memory size (MB)', fontsize=43, labelpad=30)
    ax.tick_params(axis='y', labelsize=40)

    plt.tight_layout()
    plt.savefig(f'{file_to_be_saved}.pdf', format='pdf', dpi=600)

def parse_data(filename):
    run_data_extraction(filename)

    with open(filename, 'r') as file:
        file_content = file.read()

    model_data = parse_model_requirements(file_content)
    df = pd.DataFrame(model_data).T
    df.fillna(0, inplace=True)
    
    return df, model_data

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 create_cdf.py <partitions_folder>")
        sys.exit(1)

    specific_xticks = list(range(0, 1100, 100))
    specific_xticks_show = [0, 200, 400, 600, 800, 1000]

    os.makedirs('results', exist_ok=True)

    filename = 'scripts/benchmarks/memory_requirements_detailed.txt'
    df, model_data = parse_data(filename)
    plot_model_data(df, model_data, specific_xticks, specific_xticks_show)

    filename_partitions = 'scripts/benchmarks/memory_requirements_detailed_partitions.txt'
    df, model_data = parse_data(filename_partitions)
    plot_model_data(df, model_data, specific_xticks, specific_xticks_show, is_partition=True)
    
    try:
        os.remove(filename)
        os.remove(filename_partitions)
    except FileNotFoundError:
        pass
    
if __name__ == "__main__":
    main()
