import os
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python3 create_cdf.py <partitions_folder>")
    exit(1)

associative_model_names = {
    'squeezenet1.0-7': 'SqueezeNet1.0',
    'mobilenetv2_7': 'MobileNet V2',
    'densenet-7': 'DenseNet121',
    'efficientnet-lite4-11': 'EfficientNet Lite4',
    'inception-v3-12': 'Inception V3',
    'resnet101-v2-7': 'ResNet101 V2',
    'resnet152-v2-7': 'ResNet152 V2',
    'efficientnet-v2-l-18': 'EfficientNet V2'
}

associative_model_names_partitions = {
    'squeezenet1.0-7_partitions': 'SqueezeNet1.0',
    'mobilenetv2_7_partitions': 'MobileNet V2',
    'densenet-7_partitions': 'DenseNet121',
    'efficientnet-lite4-11_partitions': 'EfficientNet Lite4',
    'inception-v3-12_partitions': 'Inception V3',
    'resnet101-v2-7_partitions': 'ResNet101 V2',
    'resnet152-v2-7_partitions': 'ResNet152 V2',
    'efficientnet-v2-l-18_partitions': 'EfficientNet V2'
}

def parse_model_requirements(text):
    model_data = {}
    
    model_blocks = text.split('Model:')
    
    for block in model_blocks[1:]:
        model_name_match = re.match(r'(\S+)\.txt', block.strip())
        if model_name_match:
            model_name = model_name_match.group(1)
            if model_name in associative_model_names:
                model_data[associative_model_names[model_name]] = {}
            elif model_name in associative_model_names_partitions:
                model_data[associative_model_names_partitions[model_name]] = {}
        
        percentage_lines = re.findall(r'exceeds ([\d.]+)MB by: ([\d.]+)%', block)
        
        for size, percentage in percentage_lines:
            percentage = float(percentage)
            if percentage == 0.0:
                percentage = 0
            if model_name in associative_model_names:
                if size in specific_xticks:
                    size = int(size)
                model_data[associative_model_names[model_name]][size] = float(percentage)
            elif model_name in associative_model_names_partitions:
                model_data[associative_model_names_partitions[model_name]][size] = float(percentage) #int(size)
    
    return model_data

def create_cdf(filename, file_to_be_saved):
    with open(filename, 'r') as file:
        file_content = file.read()
    model_data = parse_model_requirements(file_content)

    df = pd.DataFrame(model_data).T
    df.fillna(0, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    for model in df.index:
        model_data = df.loc[model]
        non_zero_data = model_data[model_data > 0]
        
        cdf = np.cumsum(non_zero_data) / non_zero_data.sum()
        ax.plot(non_zero_data.index, cdf, label=model, linewidth=2.5)

    ax.legend(fontsize=26, loc='lower right')
    ax.set_xlabel('Memory size (MB)', fontsize=30, labelpad=25)
    ax.set_ylabel('% of Total Execution Time', fontsize=30, labelpad=25)

    ax.set_xlim(left=0)
    valid_xticks = [x for x in specific_xticks if x in df.columns.astype(int)]
    ax.set_xticks(valid_xticks)
    ax.set_xticklabels(valid_xticks, fontsize=26)
    ax.set_ylim(bottom=0)
    plt.yticks(fontsize=26)

    plt.tight_layout()
    plt.savefig(file_to_be_saved + '.pdf', format='pdf', dpi=600)


if __name__ == "__main__":
    specific_xticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    if not os.path.exists('results'):
        os.mkdir('results')

    filename = 'scripts/benchmarks/memory_requirements_detailed_partitions.txt'
    if os.path.exists(filename):
        os.system(f'rm -f {filename}')
    os.system('python3 scripts/benchmarks/extract_ms_print_info.py')
    create_cdf('scripts/benchmarks/memory_requirements_detailed.txt', 'results/figure3a')

    filename = 'scripts/benchmarks/memory_requirements_detailed_partitions.txt'
    if os.path.exists(filename):
        os.system(f'rm -f {filename}')
    os.system('python3 scripts/benchmarks/extract_ms_print_info.py partitions/')
    create_cdf('scripts/benchmarks/memory_requirements_detailed_partitions.txt', 'results/figure3b')
        
    os.remove('scripts/benchmarks/memory_requirements_detailed.txt')
    os.remove('scripts/benchmarks/memory_requirements_detailed_partitions.txt')
    