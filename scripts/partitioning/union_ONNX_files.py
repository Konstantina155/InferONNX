import sys
import onnx
from onnx import helper

def merge_two_models(model1, model2):
    graph1 = model1.graph
    graph2 = model2.graph

    model1_nodes = {node.output[0]: node for node in graph1.node}
    model1_outputs = {out.name for out in graph1.output}

    input_mapping = {}
    for inp in graph2.input:
        if (inp.name in model1_outputs or inp.name in model1_nodes) and inp.name not in input_mapping:
            input_mapping[inp.name] = inp.name

    for node in graph2.node:
        node.input[:] = [input_mapping.get(i, i) for i in node.input]

    merged_inputs = []
    seen_input_names = set()
    for inp in graph1.input:
        if inp.name not in input_mapping and inp.name not in seen_input_names:
            merged_inputs.append(inp)
            seen_input_names.add(inp.name)

    for inp in graph2.input:
        if inp.name not in input_mapping and inp.name not in seen_input_names:
            merged_inputs.append(inp)
            seen_input_names.add(inp.name)

    merged_outputs = [out for out in graph2.output if out.name not in model1_outputs] + [out for out in graph1.output if out.name not in input_mapping]
    merged_initializers = list(graph1.initializer) + list(graph2.initializer)
    merged_nodes = list(graph1.node) + list(graph2.node)

    merged_graph = helper.make_graph(
        nodes=merged_nodes,
        name="merged_graph",
        inputs=merged_inputs,
        outputs=merged_outputs,
        initializer=merged_initializers
    )

    opset_import_map = {}
    for opset in model1.opset_import:
        opset_import_map[opset.domain] = opset.version
    for opset in model2.opset_import:
        if opset.domain not in opset_import_map:
            opset_import_map[opset.domain] = opset.version

    opset_imports = [helper.make_operatorsetid(domain, version)
                     for domain, version in opset_import_map.items()]

    merged_model = helper.make_model(
        merged_graph,
        producer_name=model1.producer_name,
        producer_version=model1.producer_version,
        domain=model1.domain,
        model_version=model1.model_version,
        doc_string=model1.doc_string,
        opset_imports=opset_imports,
        ir_version=model1.ir_version
    )

    return merged_model

def merge_multiple_models(models_paths, output_path):
    models_paths = list(reversed(models_paths))
    merged_model = onnx.load(models_paths[0])

    for model_path in models_paths[1:]:
        model_to_merge = onnx.load(model_path)
        merged_model = merge_two_models(merged_model, model_to_merge)

    onnx.save(merged_model, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 union_ONNX_files.py <onnx_file_#1> ... <onnx_file_#N> <output_onnx_file>")
        sys.exit(1)

    model_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]
    merged_model = merge_multiple_models(model_paths, output_path)