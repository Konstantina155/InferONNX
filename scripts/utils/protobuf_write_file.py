import sys
import onnx
import numpy as np
from onnx import numpy_helper

def extract_filename(path):
    base_name = path.split('/')[-1]
    if base_name.endswith('.onnx'):
        return base_name[:-5]
    return base_name

def write_tensor(shapes, model_name, k):
    calc_size = 1
    for number in shapes:
        calc_size *= number
    
    numpy_array = np.ones(calc_size, dtype=np.float32)
    numpy_array = np.array(numpy_array).reshape(shapes)
    tensor = numpy_helper.from_array(numpy_array)
    input_name = "../input_files/" + extract_filename(model_name) + "_" + str(k) + ".pb"
    print(input_name)
    with open(input_name, "wb") as f:
        f.write(tensor.SerializeToString())

def main():
    model_name = sys.argv[1]
    model = onnx.load(model_name)

    shapes = []
    k = 0
    for input in model.graph.input:
        shapes = []
        tensor_type = input.type.tensor_type
        if (tensor_type.HasField("shape")):
            for d in tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    shapes.append(d.dim_value)
                elif (d.HasField("dim_param")):
                    shapes.append(d.dim_param)
        write_tensor(shapes, model_name, k)
        k += 1        
        

    if shapes == []:
        write_tensor([1, 1, 1, 1], model_name, 0)

if __name__ == "__main__":
    main()