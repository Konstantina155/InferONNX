import sys
import re
from decimal import Decimal, ROUND_HALF_UP

def calculate_count(occurrence, runs):
    if occurrence == 1:
        count = 1
    else:
        count = (occurence - 1) * runs + 1
    return count

def parse_times(filename, model_name, num_partitions, runs, occurence):
    inference_times = []
    run_model_times = []
    total_server_times = []
    total_client_times = []
    partitions = [[] for _ in range(num_partitions)]
    network_times = []
    handshake = []
    handle_request = []

    count = calculate_count(occurence, runs)
    count_all = 1

    with open(filename, 'r') as file:
        lines = file.readlines()
        count_runs = 0
        for i in range(len(lines) - 1, -1, -1):
            if f"Model: {model_name}\n" in lines[i]:
                if count == count_all:
                    if lines[i+5] == "\n":
                        if "Inference time to run a model:" not in lines[i+1]:
                            stats = 5
                        else:
                            continue
                    else:
                        stats = 9
                    lines_below = lines[i:i+stats+1+num_partitions]
                    k = 0
                    
                    for index in range(len(lines_below)):
                        if num_partitions > 1 and re.search(rf"Partition_{k}: ([\d\.]+) ms", lines_below[index]) is not None:
                            partition_time = float(re.search(rf"Partition_{k}: ([\d\.]+) ms", lines_below[index]).group(1))
                            partitions[k].append(partition_time)
                            k += 1
                        if re.search(r"Inference time: ([\d\.]+) ms", lines_below[index]) is not None:
                            inference_time = float(re.search(r"Inference time: ([\d\.]+) ms", lines_below[index]).group(1))
                            inference_times.append(inference_time)
                        if re.search(r"Inference time to run a model: ([\d\.]+) ms", lines_below[index]) is not None:
                            run_model_time = float(re.search(r"Inference time to run a model: ([\d\.]+) ms", lines_below[index]).group(1))
                            run_model_times.append(run_model_time)
                        if re.search(r"Total time - server: ([\d\.]+) ms", lines_below[index]) is not None:
                            total_server_time = float(re.search(r"Total time - server: ([\d\.]+) ms", lines_below[index]).group(1))
                            total_server_times.append(total_server_time)
                        if re.search(r"Total time - client: ([\d\.]+) ms", lines_below[index]) is not None:
                            total_client_time = float(re.search(r"Total time - client: ([\d\.]+) ms", lines_below[index]).group(1))
                            total_client_times.append(total_client_time)
                            count_runs += 1
                        if re.search(r"Time to read request from client: ([\d\.]+) ms", lines_below[index]) is not None:
                            network_time = float(re.search(r"Time to read request from client: ([\d\.]+) ms", lines_below[index]).group(1))    
                        if re.search(r"Time to write response to client: ([\d\.]+) ms", lines_below[index]) is not None:
                            network_time += float(re.search(r"Time to write response to client: ([\d\.]+) ms", lines_below[index]).group(1))
                            network_times.append(network_time)
                        if re.search(r"Time to perform handshake: ([\d\.]+) ms", lines_below[index]) is not None:
                            handshake_time = float(re.search(r"Time to perform handshake: ([\d\.]+) ms", lines_below[index]).group(1))
                            handshake.append(handshake_time)
                        if re.search(r"Time to process the request: ([\d\.]+) ms", lines_below[index]) is not None:
                            handle_request_time = float(re.search(r"Time to process the request: ([\d\.]+) ms", lines_below[index]).group(1))
                            handle_request.append(handle_request_time)
                    if count_runs == runs:
                        break
                else:
                    count_all += 1
        if count_runs != runs:
            return [0], [0], [0], [0], [0], [0], [0], [[0] for _ in range(num_partitions)]
    return run_model_times, inference_times, total_server_times, total_client_times, network_times, handshake, handle_request, partitions

def calculate_average(times, runs):
    return sum(times) / runs

def convert_to_int(value):
    return Decimal(value).quantize(0, ROUND_HALF_UP)

def convert_to_float(value):
    return Decimal(value).quantize(Decimal('0.1'), ROUND_HALF_UP)

def print_results(run_model_times, inference_times, total_server_times, total_client_times, network_times, handshake, request_times, partitions):
    handshake_time = convert_to_int(calculate_average(handshake, runs))
    ssl_overhead = [0] * runs
    rest = [0] * runs

    if handshake_time == 0:
        for i in range(runs):
            rest[i] = convert_to_float(request_times[i]) - convert_to_float(inference_times[i])
            ssl_overhead[i] = convert_to_float(total_client_times[i]) - convert_to_float(request_times[i])
    else:
        for i in range(runs):
            rest[i] = convert_to_float(request_times[i]) - convert_to_float(inference_times[i])
            ssl_overhead[i] = convert_to_float(total_client_times[i]) - convert_to_float(request_times[i]) - convert_to_float(handshake[i])
            if ssl_overhead[i] < 0:
                ssl_overhead[i] = 0

    print(f"Average inference time to run a model: {convert_to_int(calculate_average(run_model_times, runs))} ms")
    print(f"Average inference time: {convert_to_int(calculate_average(inference_times, runs))} ms")
    print(f"Average total client time: {convert_to_int(calculate_average(total_client_times,runs))} ms")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 calculate_avg_time.py <file_name> <model> <number_partitions> <runs> <occurence>\n")
        sys.exit(1)

    filename = sys.argv[1]
    model_name = sys.argv[2]
    number_partitions = int(sys.argv[3])
    runs = int(sys.argv[4])
    occurence = int(sys.argv[5])
    

    run_model_times, inference_times, total_server_times, total_client_times, network_times, handshake, request, partitions = parse_times(filename, model_name, number_partitions, runs, occurence)
    print_results(run_model_times, inference_times, total_server_times, total_client_times, network_times, handshake, request, partitions)
         