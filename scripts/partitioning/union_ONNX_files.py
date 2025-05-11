import sys
import onnx
from onnx import helper

def merge_two_models(model1, model2):
    graph1, graph2 = model1.graph, model2.graph

    model1_nodes = {node.output[0]: node for node in graph1.node}
    model1_outputs = {out.name for out in graph1.output}

    input_mapping = {}
    for inp in graph2.input:
        if (inp.name in model1_outputs or inp.name in model1_nodes) and inp.name not in input_mapping:
            input_mapping[inp.name] = inp.name

    for node in graph2.node:
        node.input[:] = [input_mapping.get(i, i) for i in node.input]

    merged_inputs = []
    seen_inputs = set()
    for inp in list(graph1.input) + list(graph2.input):
        if inp.name not in input_mapping and inp.name not in seen_inputs:
            merged_inputs.append(inp)
            seen_inputs.add(inp.name)

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

    opset_import_map = {op.domain: op.version for op in model1.opset_import}
    for opset in model2.opset_import:
        opset_import_map.setdefault(opset.domain, opset.version)

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

def merge_multiple_models(model_paths, output_path):
    if not model_paths:
        raise ValueError("No models provided for merging.")
    
    models = [onnx.load(path) for path in reversed(model_paths)]
    merged_model = models[0]
    for next_model in models[1:]:
        merged_model = merge_two_models(merged_model, next_model)
    
    onnx.save(merged_model, output_path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 union_ONNX_files.py <onnx_file_#1> ... <onnx_file_#N> <output_onnx_file>")
        sys.exit(1)

    model_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]
    merge_multiple_models(model_paths, output_path)

if __name__ == "__main__":
    main()