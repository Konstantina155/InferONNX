import re

def process_stats(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    pattern = r"(Executing model .*? runs:)(.*?)(?=(Executing model|$))"
    matches = re.findall(pattern, data, re.DOTALL)

    processed_output = []

    for model_line, stats_block, _ in matches:
        lines = stats_block.strip().split('\n')
        stats = {}

        for line in lines:
            stat_match = re.match(r"Average (.+): (\d+)", line.strip())
            if stat_match:
                stat_name = stat_match.group(1).strip()
                stat_value = int(stat_match.group(2))

                # if stat_name not in ["IPC", "cache misses per instruction"]:
                #     stat_value /= 10000
                #     if stat_value.is_integer():
                #         stat_value = int(stat_value)

                stats[stat_name] = stat_value

        if "instructions" in stats and "cycles" in stats:
            stats["IPC"] = stats["instructions"] / stats["cycles"]
            stats["PCI"] = stats["cycles"] / stats["instructions"]

        if "cache_references" in stats and "cache_misses" in stats:
            stats["miss rate"] = stats["cache_misses"] / stats["cache_references"]
    
        if "cache_misses" in stats and "instructions" in stats:
            avg_cache_misses_per_instruction = stats["cache_misses"] / stats["instructions"]
            stats["cache misses per instruction"] = avg_cache_misses_per_instruction

        model_output = [model_line.strip()]
        for stat_name, value in stats.items():
            if stat_name not in ["PCI", "IPC", "cache misses per instruction", "miss rate"]:
                model_output.append(f"Average {stat_name}: {int(value)}")
            else:
                model_output.append(f"Average {stat_name}: {value:.4f}")

        processed_output.append("\n".join(model_output))

    with open(file_path, 'w') as file:
        file.write("\n\n".join(processed_output))

file_path = 'cache_stats.txt'
process_stats(file_path)