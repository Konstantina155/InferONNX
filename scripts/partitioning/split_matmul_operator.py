import os
import sys
import onnx
import subprocess
from onnx import helper, numpy_helper
from transformers import PreTrainedTokenizerFast

if len(sys.argv) != 5:
    print("python3 split_matmul_operator.py <model_name> <row/column> <number_of_divisions> <path_to_InferONNX>")
    sys.exit(1)

model_name = sys.argv[1]
split_mode = sys.argv[2]
number_of_divisions = int(sys.argv[3])
InferONNX_path = sys.argv[4]
assert number_of_divisions > 0, "Number of divisions must be > 0"
assert model_name in ["gpt2", "smol-llama-220M-GQA", "mistral-300M", "qwen2.5-0.5B"], "Model name is not correct"
assert split_mode in ["row", "column"], "Split method is not correct"

model_path = os.path.join(InferONNX_path, "models")
os.makedirs(f"{model_path}/{model_name}/original_intra_ops/", exist_ok=True)
os.makedirs(f"{model_path}/{model_name}/partitions_test/", exist_ok=True)

result = subprocess.run(f"ls {model_path}/{model_name}/partitions_inter/ | wc -w", shell=True, text=True, capture_output=True)
num_partitions = int(result.stdout.strip())

subprocess.run(f"cp {model_path}/{model_name}/partitions_inter/{model_name}_split{num_partitions - 1}.onnx {model_path}/{model_name}/original_intra_ops/", shell=True, text=True)
base_model = f"{model_path}/{model_name}/original_intra_ops/{model_name}_split{num_partitions - 1}.onnx"
path = f"{model_path}/{model_name}/partitions_test/{number_of_divisions}_parts_{split_mode}/"

# Load model and tokenizer
model = onnx.load(base_model)
graph = model.graph

### Matmul: input * weights (dot product)
matmul_input = graph.input[0]
matmul_output = graph.output[0]
matmul_weight_initializer = graph.initializer[0] # second input

tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{InferONNX_path}models/{model_name}/test_data_set_0/tokenizer.json")
print("Tokenizer vocab: ", len(tokenizer))

matmul_weight = numpy_helper.to_array(matmul_weight_initializer)
embed, total_vocab = matmul_weight.shape
print("MatMul weight shape:", matmul_weight.shape)

# Split weight matrix into {number_of_divisions} parts
os.makedirs(path, exist_ok=True)
if split_mode == "column":
    chunk = total_vocab // number_of_divisions
    weights_parts = [
        matmul_weight[:, i * chunk : (i + 1) * chunk]
        for i in range(number_of_divisions)
    ]
    if total_vocab % number_of_divisions != 0: 
        weights_parts.append(matmul_weight[:, (number_of_divisions) * chunk:])

    def create_partial_op(partial_array, index):
        weight_name = f"{matmul_weight_initializer.name}_{index}"
        output_name = f"logits_{index}"
        filename = f"{model_name}_split{num_partitions - 1}_{index}.onnx"

        weight_tensor = numpy_helper.from_array(partial_array, name=weight_name)
        
        matmul_node = helper.make_node(
            "MatMul", 
            inputs=[matmul_input.name, weight_name], 
            outputs=[output_name], 
            name=output_name
        )

        graph = helper.make_graph(
            name="main_graph_subgraph",
            nodes=[matmul_node],
            inputs=[helper.make_tensor_value_info(matmul_input.name, onnx.TensorProto.FLOAT, None)],
            outputs=[helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None)],
            initializer=[weight_tensor]
        )

        model = helper.make_model(graph, 
            ir_version=9, 
            opset_imports=[helper.make_operatorsetid("", 19)], 
            producer_name="splitter"
        )

        onnx.save(model, os.path.join(path, filename))
        
    count = 0
    if total_vocab % number_of_divisions != 0: count += 1
    for i in range(number_of_divisions + count):
        create_partial_op(weights_parts[i], i)

    # Concat the result
    concat_inputs = [f"logits_{i}" for i in range(number_of_divisions + count)]
    concat_node = helper.make_node(
        "Concat", 
        inputs=concat_inputs,
        outputs=[matmul_output.name], 
        axis=2, # concat embedding axis
        name=matmul_output.name,
    )

    concat_graph = helper.make_graph(
        name="main_graph_subgraph",
        nodes=[concat_node],
        inputs=[
            helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            for name in concat_inputs
        ],
        outputs=[helper.make_tensor_value_info(matmul_output.name, onnx.TensorProto.FLOAT, None)],
    )

    concat_model = helper.make_model(
        concat_graph, 
        ir_version=9, 
        opset_imports=[helper.make_operatorsetid("", 19)], 
        producer_name="splitter"
    )

    concat_name = f"{model_name}_split{num_partitions - 1}_{len(weights_parts)}.onnx"
    onnx.save(concat_model, os.path.join(path, concat_name))
else:
    chunk = embed // number_of_divisions
    weights_parts = [
        matmul_weight[i * chunk : (i + 1) * chunk, :]
        for i in range(number_of_divisions)
    ]
    if embed % number_of_divisions != 0: 
        weights_parts.append(matmul_weight[(number_of_divisions) * chunk:, :])

    def create_partial_op(partial_array, index):
        weight_name = f"{matmul_weight_initializer.name}_{index}"
        slice_output_name  = f"{matmul_input.name}_{index}"
        output_name = f"logits_{index}"
        filename = f"{model_name}_split{num_partitions - 1}_{index}.onnx"

        weight_tensor = numpy_helper.from_array(partial_array, name=weight_name)

        start_name = f"start_{index}"
        end_name   = f"end_{index}"
        axes_name  = f"axes_{index}"

        start_tensor = helper.make_tensor(start_name, onnx.TensorProto.INT64, [1], [index * chunk])
        end_val = (index + 1) * chunk if index < number_of_divisions else embed
        end_tensor   = helper.make_tensor(end_name,   onnx.TensorProto.INT64, [1], [end_val])
        axes_tensor  = helper.make_tensor(axes_name,  onnx.TensorProto.INT64, [1], [2])  # embedding axis
        slice_node = helper.make_node(
            "Slice",
            inputs=[matmul_input.name, start_name, end_name, axes_name],
            outputs=[slice_output_name],
            name=f"slice_{index}"
        )
        
        matmul_node = helper.make_node(
            "MatMul", 
            inputs=[slice_output_name, weight_name], 
            outputs=[output_name], 
            name=output_name
        )

        graph = helper.make_graph(
            name="main_graph_subgraph",
            nodes=[slice_node, matmul_node],
            inputs=[helper.make_tensor_value_info(matmul_input.name, onnx.TensorProto.FLOAT, None)],
            outputs=[helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None)],
            initializer=[weight_tensor, start_tensor, end_tensor, axes_tensor]
        )

        model = helper.make_model(graph, 
            ir_version=9, 
            opset_imports=[helper.make_operatorsetid("", 19)], 
            producer_name="splitter"
        )

        onnx.save(model, os.path.join(path, filename))
        
    count = 0
    if embed % number_of_divisions != 0: count += 1
    for i in range(number_of_divisions + count):
        create_partial_op(weights_parts[i], i) 

    # Add the result
    sum_inputs = [f"logits_{i}" for i in range(number_of_divisions + count)]
    sum_node = helper.make_node(
        "Sum", 
        inputs=sum_inputs,
        outputs=[matmul_output.name], 
        name=matmul_output.name,
    )

    concat_graph = helper.make_graph(
        name="main_graph_subgraph",
        nodes=[sum_node],
        inputs=[
            helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            for name in sum_inputs
        ],
        outputs=[helper.make_tensor_value_info(matmul_output.name, onnx.TensorProto.FLOAT, None)],
    )

    sum_model = helper.make_model(
        concat_graph, 
        ir_version=9, 
        opset_imports=[helper.make_operatorsetid("", 19)], 
        producer_name="splitter"
    )

    concat_name = f"{model_name}_split{num_partitions - 1}_{len(weights_parts)}.onnx"
    onnx.save(sum_model, os.path.join(path, concat_name))