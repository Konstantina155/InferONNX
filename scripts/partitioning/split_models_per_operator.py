import os
import onnx
import subprocess
from pathlib import Path
import onnx_graphsurgeon as gs

model_directory = ["squeezenet1.0-7", "mobilenetv2-7", "efficientnet-lite4-11", "resnet101-v2-7", "resnet152-v2-7", "densenet-7", "inception-v3-12", "efficientnet-v2-l-18"]

def run_command(command):
    try:
        output = subprocess.Popen(command, shell=True)
        output.wait() 
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        raise

def make_build():
    run_command("make clean")
    run_command("make")

def run_inference(directory, test_path):
    command = f"./standalone_inference {directory} {test_path}"
    try:
        output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, out_stderr = output.communicate()
        output = out_stderr.decode("utf-8")
        
        begin_of_max = output.rfind("Max is")
        if begin_of_max != -1:
            end_of_sentence = output.find("!", begin_of_max)
            if end_of_sentence != -1:
                inference_result = output[begin_of_max:end_of_sentence + 1]
                print(inference_result)
                return inference_result
    except subprocess.CalledProcessError as error:
        print(f"Inference failed: {error}")
        raise

def process_model(model_dir, path_to_store_ops):
    print(f"Processing model: {model_dir}")
    
    if path_to_store_ops.exists():
        print(f"Directory already exists: {path_to_store_ops}")
        raise FileExistsError(f"Operators directory for model {model_dir} already exists.")
    path_to_store_ops.mkdir(parents=True)

    model_name = f"../../../models/{model_dir}/{model_dir}.onnx"
    model = onnx.load(model_name)
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)

    constant_map = {output.name: node for node in graph.nodes if node.op == "Constant" for output in node.outputs}

    onnx_model = gs.export_onnx(graph)
    new_model_name = f"../../../models/{model_dir}/{model_dir}_2.onnx"
    onnx.save(onnx_model, new_model_name)

    graph = gs.import_onnx(onnx.load(new_model_name))
    i = 1
    for node in graph.nodes:
        flag = False
        if node.op == "Constant":
            continue
            
        nodes_for_partition, inputs = [node], []
        
        inputs = []
        for inp in node.inputs:
            if isinstance(inp, gs.Variable):
                if inp.name in constant_map:
                    nodes_for_partition.append(constant_map[inp.name])
                elif inp.dtype is None and inp.shape is None:
                    flag = True
                else:
                    inputs.append(inp)

        outputs = [
            out for out in node.outputs
            if isinstance(out, gs.Variable) 
            and not out.name.startswith("_") 
            and "constant" not in out.name.lower()
        ]
        
        if not inputs and not outputs:
            continue

        if flag or len(outputs) > 1:
            input_names = " ".join(inp.name for inp in inputs)
            for out in outputs:
                run_command(f"sne4onnx -if {model_name} -ion {input_names} -oon {out.name} -of {path_to_store_ops}/{model_dir}_operator{i}.onnx")
                i += 1                
        else:
            sub_graph = gs.Graph(nodes=nodes_for_partition, inputs=inputs, outputs=outputs)
            sub_model = gs.export_onnx(sub_graph)
            sub_model.opset_import[0].version = model.opset_import[0].version
            onnx.save(sub_model, f"{path_to_store_ops}/{model_dir}_operator{i}.onnx")
            i += 1

    run_command(f"rm {new_model_name}")

def main():
    curent_path = os.getcwd()
    os.chdir("src/server_with_tls/scripts/")
    make_build()

    for model_dir in model_directory:
        path_to_store_ops = Path(f"../../../models/{model_dir}/operators")
        process_model(model_dir, path_to_store_ops)

        test_path = f"../../../models/{model_dir}/test_data_set_0/input_0.pb"
        inference_operators = run_inference(f"{path_to_store_ops}/", test_path)
        inference_whole = run_inference(f"../../../models/{model_dir}/", test_path)

        if inference_operators != inference_whole:
            print(f"Operators: {inference_operators}")
            print(f"Whole: {inference_whole}")
            exit(1)
        print()

    run_command("make clean")
    os.chdir(curent_path)

if __name__ == "__main__":
    main()