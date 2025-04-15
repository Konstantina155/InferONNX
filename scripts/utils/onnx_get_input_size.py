import onnx
import sys

if len(sys.argv) != 2:
    print("Usage: python3 onnx_get_input_size.py <onnx_model>")
    exit(1)

model = onnx.load(sys.argv[1])

# The model is represented as a protobuf structure and it can be accessed
# using the standard python-for-protobuf methods

opset_version = model.opset_import[0].version
print("Opset version: ", opset_version)

size = 0
for input in model.graph.input:
    print(input.name, end=": ")
    tensor_type = input.type.tensor_type
   
    size1 = 1
    if (tensor_type.HasField("shape")):
        for d in tensor_type.shape.dim:
            if (d.HasField("dim_value")):
                print(d.dim_value, end=", ")
            elif (d.HasField("dim_param")):
                print(d.dim_param, end=", ")
            else:
                print ("?", end=", ")  # unknown dimension with no name
        size += size1
        print()
    else:
        print ("unknown rank", end="")