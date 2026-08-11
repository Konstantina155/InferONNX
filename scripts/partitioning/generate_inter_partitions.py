import os
import sys
import onnx
from onnx import utils, helper, TensorProto

def list_node_names(model_path):
    model = onnx.load(model_path)
    return [node.name for node in model.graph.node]

def create_graph():
    full_model_path = "/hdd/papafrkon/qwen2.5-0.5B/qwen2.5-0.5B.onnx"
    model = onnx.load(full_model_path, load_external_data=True)
    return model.graph, model.producer_name, model.opset_import, model.ir_version

def create_tensor_map():
    tensor_info_map = {}

    # Graph inputs
    for vi in graph.input:
        tensor_info_map[vi.name] = vi

    # Graph outputs
    for vi in graph.output:
        tensor_info_map[vi.name] = vi

    # Initializers (weights)
    for init in graph.initializer:
        vi = helper.make_tensor_value_info(init.name, init.data_type, list(init.dims))
        tensor_info_map[init.name] = vi

    return tensor_info_map

graph, producer_name, opset_import, ir_version = create_graph()
tensor_info_map = create_tensor_map()
def extract_subgraph(input_names, output_names, out_path, debug=False):
    if debug:
        print(f"Original graph has {len(graph.node)} nodes")
        print(f"Requested inputs: {input_names}")
        print(f"Requested outputs: {output_names}")
    
    # Map tensor -> producing node
    producer_map = {o: n for n in graph.node for o in n.output}
    
    if debug:
        print(f"Found {len(producer_map)} tensor->node mappings")
        # Check if requested inputs exist in the graph
        for inp in input_names:
            if inp in producer_map:
                print(f"  Input '{inp}' is produced by node: {producer_map[inp].op_type}")
            else:
                print(f"  Input '{inp}' is not produced by any node (external input)")
    
    # Identify forced inputs (inputs that are produced within the graph but we want to treat as boundaries)
    forced_inputs = set()
    natural_inputs = set()
    for inp in input_names:
        if inp in producer_map:
            forced_inputs.add(inp)
        else:
            natural_inputs.add(inp)
    
    # Create a modified producer map that excludes forced input producers
    # This prevents BFS from traversing beyond forced input points
    modified_producer_map = producer_map.copy()
    nodes_blocked_by_forced_inputs = set()
    
    if forced_inputs:
        if debug:
            print(f"Forced inputs (cutting points): {list(forced_inputs)}")
        for tensor in forced_inputs:
            if tensor in modified_producer_map:
                blocked_node = modified_producer_map[tensor]
                nodes_blocked_by_forced_inputs.add(id(blocked_node))
                # Remove from producer map so BFS stops here
                del modified_producer_map[tensor]
    
    # BFS backward from outputs to find all required nodes
    # Using modified producer map that stops at forced inputs
    required_nodes = []
    required_tensors = set(output_names)
    queue = list(output_names)
    visited_nodes = set()
    
    while queue:
        tensor = queue.pop(0)  # Use pop(0) for proper BFS
        if tensor in modified_producer_map:
            node = modified_producer_map[tensor]
            if id(node) not in visited_nodes:  # Use id() to avoid node comparison issues
                visited_nodes.add(id(node))
                required_nodes.append(node)
                # Add all inputs of this node to the queue
                for input_tensor in node.input:
                    if input_tensor and input_tensor not in required_tensors:
                        queue.append(input_tensor)
                        required_tensors.add(input_tensor)
    
    if debug:
        print(f"Found {len(required_nodes)} required nodes (after stopping at forced inputs)")
        print(f"Blocked {len(nodes_blocked_by_forced_inputs)} nodes due to forced inputs")
    
    # Collect tensors used by required nodes
    used_tensor_names = {i for n in required_nodes for i in n.input if i}
    produced_inside = {o for n in required_nodes for o in n.output if o}
    
    # Get initializer names to distinguish between weights and actual inputs
    initializer_names = {init.name for init in graph.initializer}
    
    # Real inputs = used but not produced inside subgraph + forced inputs
    # BUT exclude initializers (weights) from being treated as inputs
    dangling_inputs = used_tensor_names - produced_inside
    
    # Separate dangling inputs into actual inputs vs initializers
    dangling_actual_inputs = dangling_inputs - initializer_names
    dangling_weights = dangling_inputs & initializer_names
    
    # Real inputs = actual dangling inputs + forced inputs (but not weights)
    forced_actual_inputs = forced_inputs - initializer_names
    forced_weights = forced_inputs & initializer_names
    
    real_inputs = dangling_actual_inputs.union(forced_actual_inputs)
    
    if debug:
        print(f"Natural dangling inputs: {list(dangling_actual_inputs)}")
        print(f"Dangling weights (stay as initializers): {list(dangling_weights)}")
        print(f"Forced actual inputs: {list(forced_actual_inputs)}")
        print(f"Forced weights (stay as initializers): {list(forced_weights)}")
        print(f"Final real inputs: {list(real_inputs)}")
        print(f"Tensors produced inside: {len(produced_inside)}")
        print(f"Tensors used by subgraph: {len(used_tensor_names)}")
    
    # Check if we have any inputs / outputs
    if not real_inputs:
        raise ValueError(f"None of the specified inputs {input_names} are reachable in the subgraph")
    if not any(name in produced_inside or name in dangling_inputs for name in output_names):
        raise ValueError(f"None of the specified outputs {output_names} are produced by the subgraph")
    
    # Collect initializers (constants/weights)
    required_initializers = [init for init in graph.initializer if init.name in used_tensor_names]
    
    # Build input list
    required_inputs = []

    for name in real_inputs:
        if name in tensor_info_map:
            required_inputs.append(tensor_info_map[name])
        else:
            tensorproto = TensorProto.FLOAT
            if name == "/model/Gather_1_output_0":
                print(f"Output {name} is an UNDEFINED tensor -> INT64")
                tensorproto = TensorProto.INT64
            vi = helper.make_tensor_value_info(name, tensorproto, None)
            required_inputs.append(vi)
    
    # Build output list
    required_outputs = []
    graph_outputs_map = {vi.name: vi for vi in graph.output}
    graph_value_map   = {vi.name: vi for vi in graph.value_info}
    
    for name in output_names:
        if name in tensor_info_map:
            required_outputs.append(tensor_info_map[name])
        elif name in graph_outputs_map:
            required_outputs.append(graph_outputs_map[name])
        elif name in graph_value_map:
            required_outputs.append(graph_value_map[name])
        else:
            tensorproto = TensorProto.FLOAT
            if name == "/model/Gather_1_output_0":
                print(f"Output {name} is an UNDEFINED tensor -> INT64")
                tensorproto = TensorProto.INT64
            vi = helper.make_tensor_value_info(name, tensorproto, None)
            required_outputs.append(vi)
    
    # Build new graph (preserve original node order by not reversing)
    new_graph = helper.make_graph(
        nodes=required_nodes,  # Keep original order for better compatibility
        name=graph.name + "_subgraph",
        inputs=required_inputs,
        outputs=required_outputs,
        initializer=required_initializers,
        value_info=[v for v in graph.value_info if v.name in required_tensors],
    )
    
    # Create new model
    new_model = helper.make_model(
        new_graph,
        producer_name=producer_name,
        opset_imports=opset_import,
        ir_version=ir_version,
    )

    # Try shape inference to recover missing types
    try:
        new_model = onnx.shape_inference.infer_shapes(new_model)
        print("✅ Ran shape inference to fill in missing tensor types")
    except Exception as e:
        print(f"⚠️ Shape inference failed: {e}")
    
    # Check and save model
    try:
        onnx.checker.check_model(new_model)
        print("✅ Model passed validation checks")
    except Exception as e:
        print(f"⚠️ Model validation warning: {e}")
    
    # Save model
    onnx.save_model(new_model, out_path)
    print(f"Saved subgraph to {out_path}")
    print(f"Subgraph has {len(required_inputs)} inputs and {len(required_outputs)} outputs")
    print(f"Input names: {[inp.name for inp in required_inputs]}")
    print(f"Output names: {[out.name for out in required_outputs]}")

def split_model(model_path, input_names, output_names, out_path):    
    try:
        utils.extract_model(
            model_path,
            out_path,
            input_names=input_names,
            output_names=output_names,
        )

        model = onnx.load(out_path)
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print(f"WARNING: Saved an invalid sub-model to {out_path} that fails ONNX verification:\n   {e}")
    else:
        print(f"Saved valid sub-model to {out_path}")

def generate_partitions_gpt2(full_model_path, path_to_store_partitions):
    split_model(
        full_model_path,
        input_names=["input_ids"],
        output_names=["/transformer/Reshape_output_0", "/transformer/wpe/Gather_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split0.onnx"
    )

    # used for intra-op probably
    split_model(
        full_model_path,
        input_names=["/transformer/Reshape_output_0"],
        output_names=["/transformer/wte/Gather_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split1.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/wte/Gather_output_0", "/transformer/wpe/Gather_output_0"],
        output_names=["/transformer/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split2.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/Reshape_output_0", "attention_mask"],
        output_names=["/transformer/Mul_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split3.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/Add_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.0/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split4.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.0/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.1/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split5.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.1/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split6.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.2/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.3/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split7.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.3/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.4/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split8.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.4/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.5/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split9.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.5/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.6/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split10.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.6/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.7/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split11.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.7/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.8/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split12.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.8/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.9/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split13.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.9/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.10/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split14.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.10/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.11/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split15.onnx"
    )

    split_model(
        full_model_path,
        input_names=["input_ids", "/transformer/h.11/Add_1_output_0", "/transformer/Add_output_0"],
        output_names=["/transformer/Reshape_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split16.onnx"
    )

    # used for intra-op probably
    split_model(
        full_model_path,
        input_names=["/transformer/Reshape_3_output_0"],
        output_names=["last_hidden_state"],
        out_path=f"{path_to_store_partitions}gpt2_split17.onnx"
    )
    
    ### --- isolate MLP blocks : c_fc/Gemm, c_proj/Gemm (same MB)
    split_model(
        full_model_path,
        input_names=["input_ids"],
        output_names=["/transformer/Reshape_output_0", "/transformer/wpe/Gather_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split0.onnx"
    )

    # used for intra-op probably
    split_model(
        full_model_path,
        input_names=["/transformer/Reshape_output_0"],
        output_names=["/transformer/wte/Gather_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split1.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/wte/Gather_output_0", "/transformer/wpe/Gather_output_0"],
        output_names=["/transformer/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split2.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/Reshape_output_0", "attention_mask"],
        output_names=["/transformer/Mul_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split3.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/Add_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.0/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split4.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.0/Add_output_0"],
        output_names=["/transformer/h.0/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split5.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.0/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.0/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split6.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.0/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.0/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split7.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.0/mlp/c_fc/Gemm_output_0", "/transformer/h.0/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.0/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split8.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.0/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.0/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split9.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.0/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.0/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split10.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.0/Add_output_0", "/transformer/h.0/mlp/c_proj/Gemm_output_0", "/transformer/h.0/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.0/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split11.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.0/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.1/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split12.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.1/Add_output_0"],
        output_names=["/transformer/h.1/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split13.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.1/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.1/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split14.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.1/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.1/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split15.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.1/mlp/c_fc/Gemm_output_0", "/transformer/h.1/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.1/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split16.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.1/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.1/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split17.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.1/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.1/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split18.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.1/Add_output_0", "/transformer/h.1/mlp/c_proj/Gemm_output_0", "/transformer/h.1/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.1/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split19.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.1/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.2/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split20.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.2/Add_output_0"],
        output_names=["/transformer/h.2/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split21.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.2/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.2/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split22.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.2/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.2/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split23.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.2/mlp/c_fc/Gemm_output_0", "/transformer/h.2/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.2/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split24.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.2/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.2/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split25.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.2/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.2/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split26.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.2/Add_output_0", "/transformer/h.2/mlp/c_proj/Gemm_output_0", "/transformer/h.2/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split27.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.2/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.3/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split28.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.3/Add_output_0"],
        output_names=["/transformer/h.3/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split29.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.3/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.3/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split30.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.3/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.3/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split31.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.3/mlp/c_fc/Gemm_output_0", "/transformer/h.3/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.3/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split32.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.3/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.3/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split33.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.3/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.3/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split34.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.3/Add_output_0", "/transformer/h.3/mlp/c_proj/Gemm_output_0", "/transformer/h.3/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.3/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split35.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.3/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.4/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split36.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.4/Add_output_0"],
        output_names=["/transformer/h.4/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split37.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.4/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.4/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split38.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.4/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.4/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split39.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.4/mlp/c_fc/Gemm_output_0", "/transformer/h.4/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.4/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split40.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.4/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.4/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split41.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.4/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.4/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split42.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.4/Add_output_0", "/transformer/h.4/mlp/c_proj/Gemm_output_0", "/transformer/h.4/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.4/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split43.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.4/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.5/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split44.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.5/Add_output_0"],
        output_names=["/transformer/h.5/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split45.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.5/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.5/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split46.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.5/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.5/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split47.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.5/mlp/c_fc/Gemm_output_0", "/transformer/h.5/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.5/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split48.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.5/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.5/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split49.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.5/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.5/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split50.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.5/Add_output_0", "/transformer/h.5/mlp/c_proj/Gemm_output_0", "/transformer/h.5/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.5/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split51.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.5/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.6/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split52.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.6/Add_output_0"],
        output_names=["/transformer/h.6/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split53.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.6/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.6/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split54.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.6/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.6/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split55.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.6/mlp/c_fc/Gemm_output_0", "/transformer/h.6/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.6/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split56.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.6/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.6/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split57.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.6/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.6/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split58.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.6/Add_output_0", "/transformer/h.6/mlp/c_proj/Gemm_output_0", "/transformer/h.6/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.6/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split59.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.6/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.7/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split60.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.7/Add_output_0"],
        output_names=["/transformer/h.7/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split61.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.7/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.7/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split62.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.7/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.7/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split63.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.7/mlp/c_fc/Gemm_output_0", "/transformer/h.7/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.7/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split64.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.7/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.7/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split65.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.7/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.7/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split66.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.7/Add_output_0", "/transformer/h.7/mlp/c_proj/Gemm_output_0", "/transformer/h.7/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.7/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split67.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.7/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.8/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split68.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.8/Add_output_0"],
        output_names=["/transformer/h.8/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split69.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.8/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.8/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split70.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.8/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.8/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split71.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.8/mlp/c_fc/Gemm_output_0", "/transformer/h.8/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.8/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split72.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.8/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.8/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split73.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.8/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.8/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split74.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.8/Add_output_0", "/transformer/h.8/mlp/c_proj/Gemm_output_0", "/transformer/h.8/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.8/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split75.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.8/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.9/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split76.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.9/Add_output_0"],
        output_names=["/transformer/h.9/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split77.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.9/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.9/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split78.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.9/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.9/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split79.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.9/mlp/c_fc/Gemm_output_0", "/transformer/h.9/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.9/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split80.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.9/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.9/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split81.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.9/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.9/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split82.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.9/Add_output_0", "/transformer/h.9/mlp/c_proj/Gemm_output_0", "/transformer/h.9/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.9/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split83.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.9/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.10/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split84.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.10/Add_output_0"],
        output_names=["/transformer/h.10/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split85.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.10/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.10/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split86.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.10/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.10/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split87.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.10/mlp/c_fc/Gemm_output_0", "/transformer/h.10/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.10/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split88.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.10/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.10/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split89.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.10/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.10/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split90.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.10/Add_output_0", "/transformer/h.10/mlp/c_proj/Gemm_output_0", "/transformer/h.10/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.10/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split91.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.10/Add_1_output_0", "/transformer/Mul_output_0"],
        output_names=["/transformer/h.11/Add_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split92.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.11/Add_output_0"],
        output_names=["/transformer/h.11/ln_2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split93.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.11/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.11/mlp/c_fc/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split94.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.11/mlp/c_fc/Reshape_output_0"],
        output_names=["/transformer/h.11/mlp/c_fc/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split95.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.11/mlp/c_fc/Gemm_output_0", "/transformer/h.11/ln_2/Add_1_output_0"],
        output_names=["/transformer/h.11/mlp/act/Mul_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split96.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.11/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.11/mlp/c_proj/Reshape_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split97.onnx"
    )

    # isolate MLP block
    split_model(
        full_model_path,
        input_names=["/transformer/h.11/mlp/c_proj/Reshape_output_0"],
        output_names=["/transformer/h.11/mlp/c_proj/Gemm_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split98.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/transformer/h.11/Add_output_0", "/transformer/h.11/mlp/c_proj/Gemm_output_0", "/transformer/h.11/mlp/act/Mul_3_output_0"],
        output_names=["/transformer/h.11/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split99.onnx"
    )

    split_model(
        full_model_path,
        input_names=["input_ids", "/transformer/h.11/Add_1_output_0", "/transformer/Add_output_0"],
        output_names=["/transformer/Reshape_3_output_0"],
        out_path=f"{path_to_store_partitions}gpt2_split100.onnx"
    )

    # used for intra-op probably
    split_model(
        full_model_path,
        input_names=["/transformer/Reshape_3_output_0"],
        output_names=["last_hidden_state"],
        out_path=f"{path_to_store_partitions}gpt2_split101.onnx"
    )

def generate_partitions_llama(full_model_path, path_to_store_partitions):
    split_model(
        full_model_path,
        input_names=["input_ids"],
        output_names=["/model/embed_tokens/Gather_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split0.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/embed_tokens/Gather_output_0", "attention_mask", "position_ids"],
        output_names=["/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split1.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/embed_tokens/Gather_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.0/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split2.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/Add_output_0"],
        output_names=["/model/layers.0/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split3.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split4.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.0/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split5.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.0/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split6.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/mlp/gate_proj/MatMul_output_0", "/model/layers.0/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.0/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split7.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.0/mlp/Mul_output_0"],
        output_names=["/model/layers.0/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split8.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/post_attention_layernorm/Cast_output_0", "/model/layers.0/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.0/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split9.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.1/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split10.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/Add_output_0"],
        output_names=["/model/layers.1/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split11.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split12.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.1/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split13.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.1/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split14.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/mlp/gate_proj/MatMul_output_0", "/model/layers.1/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.1/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split15.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.1/mlp/Mul_output_0"],
        output_names=["/model/layers.1/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split16.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/post_attention_layernorm/Cast_output_0", "/model/layers.1/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.1/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split17.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.2/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split18.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/Add_output_0"],
        output_names=["/model/layers.2/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split19.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split20.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.2/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split21.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.2/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split22.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/mlp/gate_proj/MatMul_output_0", "/model/layers.2/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.2/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split23.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.2/mlp/Mul_output_0"],
        output_names=["/model/layers.2/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split24.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/post_attention_layernorm/Cast_output_0", "/model/layers.2/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split25.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.3/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split26.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/Add_output_0"],
        output_names=["/model/layers.3/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split27.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split28.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.3/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split29.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.3/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split30.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/mlp/gate_proj/MatMul_output_0", "/model/layers.3/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.3/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split31.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.3/mlp/Mul_output_0"],
        output_names=["/model/layers.3/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split32.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/post_attention_layernorm/Cast_output_0", "/model/layers.3/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.3/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split33.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.4/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split34.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/Add_output_0"],
        output_names=["/model/layers.4/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split35.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split36.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.4/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split37.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.4/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split38.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/mlp/gate_proj/MatMul_output_0", "/model/layers.4/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.4/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split39.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.4/mlp/Mul_output_0"],
        output_names=["/model/layers.4/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split40.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/post_attention_layernorm/Cast_output_0", "/model/layers.4/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.4/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split41.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.5/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split42.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/Add_output_0"],
        output_names=["/model/layers.5/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split43.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split44.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.5/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split45.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.5/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split46.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/mlp/gate_proj/MatMul_output_0", "/model/layers.5/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.5/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split47.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.5/mlp/Mul_output_0"],
        output_names=["/model/layers.5/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split48.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/post_attention_layernorm/Cast_output_0", "/model/layers.5/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.5/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split49.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.6/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split50.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/Add_output_0"],
        output_names=["/model/layers.6/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split51.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split52.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.6/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split53.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.6/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split54.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/mlp/gate_proj/MatMul_output_0", "/model/layers.6/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.6/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split55.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.6/mlp/Mul_output_0"],
        output_names=["/model/layers.6/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split56.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/post_attention_layernorm/Cast_output_0", "/model/layers.6/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.6/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split57.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.7/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split58.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/Add_output_0"],
        output_names=["/model/layers.7/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split59.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split60.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.7/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split61.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.7/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split62.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/mlp/gate_proj/MatMul_output_0", "/model/layers.7/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.7/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split63.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.7/mlp/Mul_output_0"],
        output_names=["/model/layers.7/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split64.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/post_attention_layernorm/Cast_output_0", "/model/layers.7/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.7/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split65.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.8/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split66.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/Add_output_0"],
        output_names=["/model/layers.8/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split67.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split68.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.8/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split69.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.8/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split70.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/mlp/gate_proj/MatMul_output_0", "/model/layers.8/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.8/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split71.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.8/mlp/Mul_output_0"],
        output_names=["/model/layers.8/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split72.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/post_attention_layernorm/Cast_output_0", "/model/layers.8/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.8/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split73.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.9/Add_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split74.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/Add_output_0"],
        output_names=["/model/layers.9/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split75.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split76.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.9/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split77.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.9/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split78.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/mlp/gate_proj/MatMul_output_0", "/model/layers.9/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.9/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split79.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.9/mlp/Mul_output_0"],
        output_names=["/model/layers.9/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split80.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/post_attention_layernorm/Cast_output_0", "/model/layers.9/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.9/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split81.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/Add_1_output_0"],
        output_names=["/model/norm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split82.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/norm/Mul_1_output_0"],
        output_names=["logits"],
        out_path=f"{path_to_store_partitions}smol-llama-220M-GQA_split83.onnx"
    )     

def generate_partitions_mistral(full_model_path, path_to_store_partitions):
    split_model(
        full_model_path,
        input_names=["input_ids"],
        output_names=["/model/embed_tokens/Gather_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split0.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/embed_tokens/Gather_output_0", "attention_mask", "position_ids"],
        output_names=["/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split1.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/embed_tokens/Gather_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.0/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split2.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/Add_output_0"],
        output_names=["/model/layers.0/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split3.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split4.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.0/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split5.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.0/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split6.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/mlp/gate_proj/MatMul_output_0", "/model/layers.0/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.0/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split7.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.0/mlp/Mul_output_0"],
        output_names=["/model/layers.0/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split8.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/post_attention_layernorm/Cast_output_0", "/model/layers.0/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.0/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split9.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.0/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.1/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split10.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/Add_output_0"],
        output_names=["/model/layers.1/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split11.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split12.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.1/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split13.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.1/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split14.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/mlp/gate_proj/MatMul_output_0", "/model/layers.1/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.1/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split15.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.1/mlp/Mul_output_0"],
        output_names=["/model/layers.1/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split16.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/post_attention_layernorm/Cast_output_0", "/model/layers.1/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.1/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split17.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.1/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.2/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split18.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/Add_output_0"],
        output_names=["/model/layers.2/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split19.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split20.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.2/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split21.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.2/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split22.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/mlp/gate_proj/MatMul_output_0", "/model/layers.2/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.2/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split23.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.2/mlp/Mul_output_0"],
        output_names=["/model/layers.2/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split24.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/post_attention_layernorm/Cast_output_0", "/model/layers.2/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split25.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.2/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.3/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split26.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/Add_output_0"],
        output_names=["/model/layers.3/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split27.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split28.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.3/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split29.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.3/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split30.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/mlp/gate_proj/MatMul_output_0", "/model/layers.3/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.3/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split31.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.3/mlp/Mul_output_0"],
        output_names=["/model/layers.3/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split32.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/post_attention_layernorm/Cast_output_0", "/model/layers.3/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.3/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split33.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.3/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.4/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split34.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/Add_output_0"],
        output_names=["/model/layers.4/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split35.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split36.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.4/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split37.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.4/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split38.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/mlp/gate_proj/MatMul_output_0", "/model/layers.4/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.4/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split39.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.4/mlp/Mul_output_0"],
        output_names=["/model/layers.4/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split40.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/post_attention_layernorm/Cast_output_0", "/model/layers.4/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.4/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split41.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.4/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.5/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split42.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/Add_output_0"],
        output_names=["/model/layers.5/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split43.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split44.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.5/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split45.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.5/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split46.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/mlp/gate_proj/MatMul_output_0", "/model/layers.5/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.5/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split47.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.5/mlp/Mul_output_0"],
        output_names=["/model/layers.5/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split48.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/post_attention_layernorm/Cast_output_0", "/model/layers.5/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.5/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split49.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.5/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.6/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split50.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/Add_output_0"],
        output_names=["/model/layers.6/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split51.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split52.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.6/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split53.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.6/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split54.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/mlp/gate_proj/MatMul_output_0", "/model/layers.6/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.6/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split55.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.6/mlp/Mul_output_0"],
        output_names=["/model/layers.6/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split56.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/post_attention_layernorm/Cast_output_0", "/model/layers.6/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.6/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split57.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.6/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.7/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split58.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/Add_output_0"],
        output_names=["/model/layers.7/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split59.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split60.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.7/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split61.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.7/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split62.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/mlp/gate_proj/MatMul_output_0", "/model/layers.7/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.7/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split63.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.7/mlp/Mul_output_0"],
        output_names=["/model/layers.7/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split64.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/post_attention_layernorm/Cast_output_0", "/model/layers.7/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.7/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split65.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.7/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.8/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split66.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/Add_output_0"],
        output_names=["/model/layers.8/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split67.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split68.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.8/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split69.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.8/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split70.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/mlp/gate_proj/MatMul_output_0", "/model/layers.8/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.8/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split71.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.8/mlp/Mul_output_0"],
        output_names=["/model/layers.8/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split72.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/post_attention_layernorm/Cast_output_0", "/model/layers.8/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.8/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split73.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.8/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.9/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split74.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/Add_output_0"],
        output_names=["/model/layers.9/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split75.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split76.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.9/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split77.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.9/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split78.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/mlp/gate_proj/MatMul_output_0", "/model/layers.9/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.9/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split79.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.9/mlp/Mul_output_0"],
        output_names=["/model/layers.9/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split80.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/post_attention_layernorm/Cast_output_0", "/model/layers.9/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.9/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split81.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.9/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.10/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split82.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.10/Add_output_0"],
        output_names=["/model/layers.10/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split83.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.10/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.10/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split84.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.10/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.10/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split85.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.10/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.10/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split86.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.10/mlp/gate_proj/MatMul_output_0", "/model/layers.10/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.10/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split87.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.10/mlp/Mul_output_0"],
        output_names=["/model/layers.10/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split88.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.10/post_attention_layernorm/Cast_output_0", "/model/layers.10/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.10/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split89.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.10/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.11/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split90.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.11/Add_output_0"],
        output_names=["/model/layers.11/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split91.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.11/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.11/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split92.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.11/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.11/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split93.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.11/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.11/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split94.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.11/mlp/gate_proj/MatMul_output_0", "/model/layers.11/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.11/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split95.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.11/mlp/Mul_output_0"],
        output_names=["/model/layers.11/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split96.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.11/post_attention_layernorm/Cast_output_0", "/model/layers.11/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.11/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split97.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.11/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.12/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split98.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.12/Add_output_0"],
        output_names=["/model/layers.12/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split99.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.12/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.12/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split100.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.12/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.12/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split101.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.12/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.12/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split102.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.12/mlp/gate_proj/MatMul_output_0", "/model/layers.12/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.12/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split103.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.12/mlp/Mul_output_0"],
        output_names=["/model/layers.12/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split104.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.12/post_attention_layernorm/Cast_output_0", "/model/layers.12/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.12/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split105.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.12/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.13/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split106.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.13/Add_output_0"],
        output_names=["/model/layers.13/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split107.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.13/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.13/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split108.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.13/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.13/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split109.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.13/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.13/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split110.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.13/mlp/gate_proj/MatMul_output_0", "/model/layers.13/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.13/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split111.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.13/mlp/Mul_output_0"],
        output_names=["/model/layers.13/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split112.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.13/post_attention_layernorm/Cast_output_0", "/model/layers.13/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.13/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split113.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.13/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.14/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split114.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.14/Add_output_0"],
        output_names=["/model/layers.14/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split115.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.14/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.14/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split116.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.14/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.14/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split117.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.14/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.14/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split118.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.14/mlp/gate_proj/MatMul_output_0", "/model/layers.14/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.14/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split119.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.14/mlp/Mul_output_0"],
        output_names=["/model/layers.14/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split120.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.14/post_attention_layernorm/Cast_output_0", "/model/layers.14/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.14/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split121.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.14/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.15/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split122.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.15/Add_output_0"],
        output_names=["/model/layers.15/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split123.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.15/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.15/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split124.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.15/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.15/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split125.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.15/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.15/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split126.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.15/mlp/gate_proj/MatMul_output_0", "/model/layers.15/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.15/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split127.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.15/mlp/Mul_output_0"],
        output_names=["/model/layers.15/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split128.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.15/post_attention_layernorm/Cast_output_0", "/model/layers.15/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.15/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split129.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.15/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.16/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split130.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.16/Add_output_0"],
        output_names=["/model/layers.16/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split131.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.16/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.16/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split132.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.16/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.16/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split133.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.16/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.16/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split134.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.16/mlp/gate_proj/MatMul_output_0", "/model/layers.16/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.16/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split135.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.16/mlp/Mul_output_0"],
        output_names=["/model/layers.16/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split136.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.16/post_attention_layernorm/Cast_output_0", "/model/layers.16/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.16/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split137.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.16/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.17/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split138.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.17/Add_output_0"],
        output_names=["/model/layers.17/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split139.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.17/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.17/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split140.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.17/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.17/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split141.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.17/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.17/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split142.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.17/mlp/gate_proj/MatMul_output_0", "/model/layers.17/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.17/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split143.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.17/mlp/Mul_output_0"],
        output_names=["/model/layers.17/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split144.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.17/post_attention_layernorm/Cast_output_0", "/model/layers.17/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.17/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split145.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.17/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.18/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split146.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.18/Add_output_0"],
        output_names=["/model/layers.18/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split147.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.18/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.18/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split148.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.18/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.18/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split149.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.18/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.18/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split150.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.18/mlp/gate_proj/MatMul_output_0", "/model/layers.18/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.18/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split151.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.18/mlp/Mul_output_0"],
        output_names=["/model/layers.18/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split152.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.18/post_attention_layernorm/Cast_output_0", "/model/layers.18/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.18/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split153.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.18/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.19/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split154.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.19/Add_output_0"],
        output_names=["/model/layers.19/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split155.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.19/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.19/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split156.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.19/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.19/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split157.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.19/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.19/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split158.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.19/mlp/gate_proj/MatMul_output_0", "/model/layers.19/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.19/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split159.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.19/mlp/Mul_output_0"],
        output_names=["/model/layers.19/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split160.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.19/post_attention_layernorm/Cast_output_0", "/model/layers.19/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.19/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split161.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.19/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.20/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split162.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.20/Add_output_0"],
        output_names=["/model/layers.20/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split163.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.20/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.20/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split164.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.20/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.20/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split165.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.20/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.20/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split166.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.20/mlp/gate_proj/MatMul_output_0", "/model/layers.20/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.20/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split167.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.20/mlp/Mul_output_0"],
        output_names=["/model/layers.20/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split168.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.20/post_attention_layernorm/Cast_output_0", "/model/layers.20/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.20/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split169.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.20/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.21/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split170.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.21/Add_output_0"],
        output_names=["/model/layers.21/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split171.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.21/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.21/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split172.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.21/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.21/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split173.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.21/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.21/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split174.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.21/mlp/gate_proj/MatMul_output_0", "/model/layers.21/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.21/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split175.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.21/mlp/Mul_output_0"],
        output_names=["/model/layers.21/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split176.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.21/post_attention_layernorm/Cast_output_0", "/model/layers.21/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.21/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split177.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.21/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.22/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split178.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.22/Add_output_0"],
        output_names=["/model/layers.22/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split179.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.22/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.22/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split180.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.22/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.22/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split181.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.22/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.22/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split182.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.22/mlp/gate_proj/MatMul_output_0", "/model/layers.22/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.22/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split183.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.22/mlp/Mul_output_0"],
        output_names=["/model/layers.22/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split184.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.22/post_attention_layernorm/Cast_output_0", "/model/layers.22/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.22/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split185.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.22/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.23/Add_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split186.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.23/Add_output_0"],
        output_names=["/model/layers.23/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split187.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.23/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.23/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split188.onnx"
    )

    # first heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.23/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.23/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split189.onnx"
    )

    # second heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.23/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.23/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split190.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.23/mlp/gate_proj/MatMul_output_0", "/model/layers.23/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.23/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split191.onnx"
    )

    # third heavy op
    split_model(
        full_model_path,
        input_names=["/model/layers.23/mlp/Mul_output_0"],
        output_names=["/model/layers.23/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split192.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.23/post_attention_layernorm/Cast_output_0", "/model/layers.23/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.23/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split193.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/layers.23/Add_1_output_0"],
        output_names=["/model/norm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}mistral-300M_split194.onnx"
    )

    split_model(
        full_model_path,
        input_names=["/model/norm/Mul_1_output_0"],
        output_names=["logits"],
        out_path=f"{path_to_store_partitions}mistral-300M_split195.onnx"
    )

def generate_partitions_qwen(full_model_path, path_to_store_partitions):
    extract_subgraph(
        input_names = ["input_ids"],
        output_names = ["/model/embed_tokens/Gather_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split0.onnx",
        debug=True
    )

    extract_subgraph(
        input_names = ["attention_mask", "position_ids", "/model/embed_tokens/Gather_output_0"],
        output_names = ["/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split1.onnx",
        debug=True
    )
    
    extract_subgraph(
        input_names = ["/model/embed_tokens/Gather_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names = ["/model/layers.0/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split2.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.0/Add_output_0"],
        output_names=["/model/layers.0/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split3.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.0/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split4.onnx",
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.0/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split5.onnx",
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.0/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.0/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split6.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.0/mlp/gate_proj/MatMul_output_0", "/model/layers.0/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.0/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split7.onnx",
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.0/mlp/Mul_output_0"],
        output_names=["/model/layers.0/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split8.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.0/post_attention_layernorm/Cast_output_0", "/model/layers.0/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.0/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split9.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.0/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.1/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split10.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.1/Add_output_0"],
        output_names=["/model/layers.1/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split11.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.1/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split12.onnx",
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.1/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split13.onnx",
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.1/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.1/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split14.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.1/mlp/gate_proj/MatMul_output_0", "/model/layers.1/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.1/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split15.onnx",
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.1/mlp/Mul_output_0"],
        output_names=["/model/layers.1/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split16.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.1/post_attention_layernorm/Cast_output_0", "/model/layers.1/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.1/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split17.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.1/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.2/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split18.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.2/Add_output_0"],
        output_names=["/model/layers.2/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split19.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.2/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split20.onnx",
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.2/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split21.onnx",
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.2/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.2/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split22.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.2/mlp/gate_proj/MatMul_output_0", "/model/layers.2/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.2/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split23.onnx",
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.2/mlp/Mul_output_0"],
        output_names=["/model/layers.2/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split24.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.2/post_attention_layernorm/Cast_output_0", "/model/layers.2/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.2/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split25.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.2/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.3/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split26.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.3/Add_output_0"],
        output_names=["/model/layers.3/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split27.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.3/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split28.onnx",
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.3/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split29.onnx",
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.3/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.3/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split30.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.3/mlp/gate_proj/MatMul_output_0", "/model/layers.3/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.3/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split31.onnx",
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.3/mlp/Mul_output_0"],
        output_names=["/model/layers.3/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split32.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.3/post_attention_layernorm/Cast_output_0", "/model/layers.3/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.3/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split33.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.3/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.4/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split34.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.4/Add_output_0"],
        output_names=["/model/layers.4/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split35.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.4/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split36.onnx",
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.4/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split37.onnx",
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.4/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.4/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split38.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.4/mlp/gate_proj/MatMul_output_0", "/model/layers.4/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.4/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split39.onnx",
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.4/mlp/Mul_output_0"],
        output_names=["/model/layers.4/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split40.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.4/post_attention_layernorm/Cast_output_0", "/model/layers.4/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.4/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split41.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.4/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.5/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split42.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.5/Add_output_0"],
        output_names=["/model/layers.5/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split43.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.5/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split44.onnx",
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.5/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split45.onnx",
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.5/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.5/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split46.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.5/mlp/gate_proj/MatMul_output_0", "/model/layers.5/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.5/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split47.onnx", 
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.5/mlp/Mul_output_0"],
        output_names=["/model/layers.5/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split48.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.5/post_attention_layernorm/Cast_output_0", "/model/layers.5/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.5/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split49.onnx",
    )

    extract_subgraph(
        input_names=["/model/layers.5/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.6/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split50.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.6/Add_output_0"],
        output_names=["/model/layers.6/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split51.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.6/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split52.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.6/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split53.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.6/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.6/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split54.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.6/mlp/gate_proj/MatMul_output_0", "/model/layers.6/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.6/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split55.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.6/mlp/Mul_output_0"],
        output_names=["/model/layers.6/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split56.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.6/post_attention_layernorm/Cast_output_0", "/model/layers.6/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.6/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split57.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.6/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.7/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split58.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.7/Add_output_0"],
        output_names=["/model/layers.7/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split59.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.7/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split60.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.7/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split61.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.7/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.7/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split62.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.7/mlp/gate_proj/MatMul_output_0", "/model/layers.7/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.7/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split63.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.7/mlp/Mul_output_0"],
        output_names=["/model/layers.7/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split64.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.7/post_attention_layernorm/Cast_output_0", "/model/layers.7/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.7/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split65.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.7/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.8/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split66.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.8/Add_output_0"],
        output_names=["/model/layers.8/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split67.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.8/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split68.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.8/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split69.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.8/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.8/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split70.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.8/mlp/gate_proj/MatMul_output_0", "/model/layers.8/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.8/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split71.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.8/mlp/Mul_output_0"],
        output_names=["/model/layers.8/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split72.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.8/post_attention_layernorm/Cast_output_0", "/model/layers.8/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.8/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split73.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.8/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.9/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split74.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.9/Add_output_0"],
        output_names=["/model/layers.9/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split75.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.9/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split76.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.9/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split77.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.9/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.9/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split78.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.9/mlp/gate_proj/MatMul_output_0", "/model/layers.9/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.9/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split79.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.9/mlp/Mul_output_0"],
        output_names=["/model/layers.9/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split80.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.9/post_attention_layernorm/Cast_output_0", "/model/layers.9/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.9/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split81.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.9/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.10/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split82.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.10/Add_output_0"],
        output_names=["/model/layers.10/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split83.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.10/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.10/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split84.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.10/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.10/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split85.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.10/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.10/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split86.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.10/mlp/gate_proj/MatMul_output_0", "/model/layers.10/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.10/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split87.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.10/mlp/Mul_output_0"],
        output_names=["/model/layers.10/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split88.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.10/post_attention_layernorm/Cast_output_0", "/model/layers.10/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.10/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split89.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.10/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.11/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split90.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.11/Add_output_0"],
        output_names=["/model/layers.11/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split91.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.11/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.11/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split92.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.11/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.11/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split93.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.11/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.11/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split94.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.11/mlp/gate_proj/MatMul_output_0", "/model/layers.11/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.11/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split95.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.11/mlp/Mul_output_0"],
        output_names=["/model/layers.11/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split96.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.11/post_attention_layernorm/Cast_output_0", "/model/layers.11/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.11/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split97.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.11/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.12/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split98.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.12/Add_output_0"],
        output_names=["/model/layers.12/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split99.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.12/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.12/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split100.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.12/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.12/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split101.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.12/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.12/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split102.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.12/mlp/gate_proj/MatMul_output_0", "/model/layers.12/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.12/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split103.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.12/mlp/Mul_output_0"],
        output_names=["/model/layers.12/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split104.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.12/post_attention_layernorm/Cast_output_0", "/model/layers.12/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.12/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split105.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.12/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.13/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split106.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.13/Add_output_0"],
        output_names=["/model/layers.13/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split107.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.13/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.13/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split108.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.13/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.13/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split109.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.13/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.13/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split110.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.13/mlp/gate_proj/MatMul_output_0", "/model/layers.13/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.13/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split111.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.13/mlp/Mul_output_0"],
        output_names=["/model/layers.13/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split112.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.13/post_attention_layernorm/Cast_output_0", "/model/layers.13/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.13/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split113.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.13/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.14/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split114.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.14/Add_output_0"],
        output_names=["/model/layers.14/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split115.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.14/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.14/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split116.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.14/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.14/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split117.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.14/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.14/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split118.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.14/mlp/gate_proj/MatMul_output_0", "/model/layers.14/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.14/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split119.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.14/mlp/Mul_output_0"],
        output_names=["/model/layers.14/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split120.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.14/post_attention_layernorm/Cast_output_0", "/model/layers.14/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.14/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split121.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.14/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.15/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split122.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.15/Add_output_0"],
        output_names=["/model/layers.15/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split123.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.15/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.15/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split124.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.15/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.15/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split125.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.15/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.15/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split126.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.15/mlp/gate_proj/MatMul_output_0", "/model/layers.15/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.15/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split127.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.15/mlp/Mul_output_0"],
        output_names=["/model/layers.15/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split128.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.15/post_attention_layernorm/Cast_output_0", "/model/layers.15/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.15/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split129.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.15/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.16/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split130.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.16/Add_output_0"],
        output_names=["/model/layers.16/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split131.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.16/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.16/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split132.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.16/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.16/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split133.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.16/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.16/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split134.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.16/mlp/gate_proj/MatMul_output_0", "/model/layers.16/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.16/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split135.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.16/mlp/Mul_output_0"],
        output_names=["/model/layers.16/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split136.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.16/post_attention_layernorm/Cast_output_0", "/model/layers.16/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.16/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split137.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.16/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.17/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split138.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.17/Add_output_0"],
        output_names=["/model/layers.17/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split139.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.17/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.17/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split140.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.17/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.17/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split141.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.17/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.17/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split142.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.17/mlp/gate_proj/MatMul_output_0", "/model/layers.17/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.17/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split143.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.17/mlp/Mul_output_0"],
        output_names=["/model/layers.17/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split144.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.17/post_attention_layernorm/Cast_output_0", "/model/layers.17/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.17/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split145.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.17/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.18/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split146.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.18/Add_output_0"],
        output_names=["/model/layers.18/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split147.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.18/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.18/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split148.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.18/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.18/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split149.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.18/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.18/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split150.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.18/mlp/gate_proj/MatMul_output_0", "/model/layers.18/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.18/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split151.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.18/mlp/Mul_output_0"],
        output_names=["/model/layers.18/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split152.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.18/post_attention_layernorm/Cast_output_0", "/model/layers.18/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.18/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split153.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.18/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.19/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split154.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.19/Add_output_0"],
        output_names=["/model/layers.19/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split155.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.19/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.19/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split156.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.19/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.19/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split157.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.19/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.19/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split158.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.19/mlp/gate_proj/MatMul_output_0", "/model/layers.19/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.19/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split159.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.19/mlp/Mul_output_0"],
        output_names=["/model/layers.19/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split160.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.19/post_attention_layernorm/Cast_output_0", "/model/layers.19/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.19/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split161.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.19/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.20/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split162.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.20/Add_output_0"],
        output_names=["/model/layers.20/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split163.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.20/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.20/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split164.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.20/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.20/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split165.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.20/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.20/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split166.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.20/mlp/gate_proj/MatMul_output_0", "/model/layers.20/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.20/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split167.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.20/mlp/Mul_output_0"],
        output_names=["/model/layers.20/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split168.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.20/post_attention_layernorm/Cast_output_0", "/model/layers.20/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.20/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split169.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.20/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.21/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split170.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.21/Add_output_0"],
        output_names=["/model/layers.21/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split171.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.21/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.21/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split172.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.21/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.21/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split173.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.21/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.21/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split174.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.21/mlp/gate_proj/MatMul_output_0", "/model/layers.21/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.21/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split175.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.21/mlp/Mul_output_0"],
        output_names=["/model/layers.21/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split176.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.21/post_attention_layernorm/Cast_output_0", "/model/layers.21/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.21/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split177.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.21/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.22/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split178.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.22/Add_output_0"],
        output_names=["/model/layers.22/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split179.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.22/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.22/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split180.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.22/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.22/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split181.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.22/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.22/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split182.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.22/mlp/gate_proj/MatMul_output_0", "/model/layers.22/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.22/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split183.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.22/mlp/Mul_output_0"],
        output_names=["/model/layers.22/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split184.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.22/post_attention_layernorm/Cast_output_0", "/model/layers.22/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.22/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split185.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.22/Add_1_output_0", "/model/ScatterND_output_0", "/model/layers.0/self_attn/Unsqueeze_6_output_0", "/model/layers.0/self_attn/Unsqueeze_7_output_0"],
        output_names=["/model/layers.23/Add_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split186.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.23/Add_output_0"],
        output_names=["/model/layers.23/post_attention_layernorm/Cast_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split187.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.23/post_attention_layernorm/Cast_output_0"],
        output_names=["/model/layers.23/post_attention_layernorm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split188.onnx"
    )

    # first heavy op
    extract_subgraph(
        input_names=["/model/layers.23/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.23/mlp/gate_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split189.onnx"
    )

    # second heavy op
    extract_subgraph(
        input_names=["/model/layers.23/post_attention_layernorm/Mul_1_output_0"],
        output_names=["/model/layers.23/mlp/up_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split190.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.23/mlp/gate_proj/MatMul_output_0", "/model/layers.23/mlp/up_proj/MatMul_output_0"],
        output_names=["/model/layers.23/mlp/Mul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split191.onnx"
    )

    # third heavy op
    extract_subgraph(
        input_names=["/model/layers.23/mlp/Mul_output_0"],
        output_names=["/model/layers.23/mlp/down_proj/MatMul_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split192.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.23/post_attention_layernorm/Cast_output_0", "/model/layers.23/mlp/down_proj/MatMul_output_0"],
        output_names=["/model/layers.23/Add_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split193.onnx"
    )

    extract_subgraph(
        input_names=["/model/layers.23/Add_1_output_0"],
        output_names=["/model/norm/Mul_1_output_0"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split194.onnx"
    )

    extract_subgraph(
        input_names=["/model/norm/Mul_1_output_0"],
        output_names=["logits"],
        out_path=f"{path_to_store_partitions}qwen2.5-0.5B_split195.onnx"
    )

def main():
    if len(sys.argv) != 1:
        print("Usage: python3 generate_inter_partitions.py\n")
        exit(1)

    models = ["gpt2", "smol-llama-220M-GQA", "mistral-300M", "qwen2.5-0.5B"]
    for model in models:
        full_model_path = f"./models/{model}/{model}.onnx"
        path_to_store_partitions = f"./models/{model}/partitions_inter/"
        os.makedirs(path_to_store_partitions, exist_ok=True)

        ###---------STEP 1: List node names (here or observe from Netron)---------###
        node_names = list_node_names(full_model_path)

        ###---------STEP 2: Partition models---------###
        if "gpt" in model:
            generate_partitions_gpt2(full_model_path, path_to_store_partitions)
        elif "llama" in model:
            generate_partitions_llama(full_model_path, path_to_store_partitions)
        elif "mistral" in model:
            generate_partitions_mistral(full_model_path, path_to_store_partitions)
        else:
            generate_partitions_qwen(full_model_path, path_to_store_partitions)

if __name__ == "__main__":
    main()