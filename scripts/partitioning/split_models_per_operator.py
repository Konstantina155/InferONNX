import onnx_graphsurgeon as gs
import numpy as np
import subprocess
import networkx as nx
import onnx
import os
import sys

model_directory = ["squeezenet1.0-7", "mobilenetv2-7", "efficientnet-lite4-11", "resnet101-v2-7", "resnet152-v2-7", "densenet-7", "inception-v3-12", "efficientnet-v2-l-18"]

def run_inference(directory, test_path):
    command = f"./standalone_inference {directory} {test_path}"
    try:
        output = subprocess.run(command, shell=True, stderr=subprocess.PIPE, check=True)
        output = output.stderr.decode("utf-8")
        
        beginning_of_max = output.rfind("Max is")
        if beginning_of_max != -1:
            end_of_sentence = output.find("!", beginning_of_max)
            if end_of_sentence != -1:
                print(output[beginning_of_max:end_of_sentence + 1])

        return output[beginning_of_max:end_of_sentence + 1]
    except subprocess.CalledProcessError as error:
        print(error)
        exit(1)

def main():
    curent_path = os.getcwd()
    os.chdir("src/server_with_tls/scripts/")
    os.system("make clean && make")

    for model_dir in model_directory:
        path_to_store_operators = "../../../models/" + model_dir + "/operators"
 
        print("Running for model: ", model_dir)


        if not os.path.exists(path_to_store_operators):
            os.system(f"mkdir {path_to_store_operators}")
        else:
            print(f"Directory {path_to_store_operators} already exists")
            exit(1)

        model_name = "../../../models/" + model_dir + "/" + model_dir + ".onnx"
        model = onnx.load(model_name)
        model = onnx.shape_inference.infer_shapes(model)
        graph = gs.import_onnx(model)

        constant_map = {}
        for node in graph.nodes:
            if node.op == "Constant":
                for output in node.outputs:
                    constant_map[output.name] = node

        onnx_model = gs.export_onnx(graph)
        new_model_name = "../../../models/" + model_dir + "/" + model_dir + "_2.onnx"
        onnx.save(onnx_model, new_model_name)
        model = onnx.load(new_model_name)
        graph = gs.import_onnx(model)

        i = 1
        for node in graph.nodes:
            print(node)
            flag = False

            if node.op == "Constant":
                continue
                
            nodes_for_partition = [node]
            
            inputs = []
            for inp in node.inputs:
                if isinstance(inp, gs.Variable):
                    if inp.name in constant_map:
                        nodes_for_partition.append(constant_map[inp.name])
                    else:
                        if inp.dtype is None and inp.shape is None:
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

            if len(outputs) > 1:
                flag = True

            if flag:
                input_names = " ".join(inp.name for inp in inputs)
                for out in outputs:
                    os.system(f"sne4onnx -if {model_name} -ion {input_names} -oon {out.name} -of {path_to_store_operators}/{model_dir}_operator{i}.onnx")
                    i += 1                
            else:
                sub_graph = gs.Graph(nodes=nodes_for_partition, inputs=inputs, outputs=outputs)
                sub_model = gs.export_onnx(sub_graph)
                sub_model.opset_import[0].version = model.opset_import[0].version
                onnx.save(sub_model, f"{path_to_store_operators}/{model_dir}_operator{i}.onnx")
                i += 1

        os.system("rm " + new_model_name)

        test_path = f"../../../models/{model_dir}/test_data_set_0/input_0.pb"
        inference_operators = run_inference(f"{path_to_store_operators}/", test_path)
        inference_whole = run_inference(f"../../../models/{model_dir}/", test_path)

        if inference_operators != inference_whole:
            print(f"Operators: {inference_operators}")
            print(f"Whole: {inference_whole}")
            exit(1)
        print()

    os.system("make clean && make")
    os.chdir(curent_path)

if __name__ == "__main__":
    main()