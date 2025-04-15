import onnx
import sys
from onnx import numpy_helper
import numpy
import subprocess
import ast

if len(sys.argv) != 2:
    print("Usage: {} <model>".format(sys.argv[0]))
    sys.exit(1)

def extract_filename(path):
    base_name = path.split('/')[-1]
    if base_name.endswith('.onnx'):
        return base_name[:-5]
    return base_name

def extract_inputs(lines):
    for line in lines:
        if line.startswith("Inputs:"):
            return ast.literal_eval(line.split(":", 1)[1].strip())
    return []

model_name = sys.argv[1]
model = onnx.load(model_name)
output = subprocess.run(f"python3 scripts/utils/onnx_get_io.py {model_name}".format(sys.argv[1]), shell=True, check=True, text=True, capture_output=True)
inputs_list = extract_inputs(output.stdout.split("\n"))
output = subprocess.run(f"python3 scripts/utils/onnx_get_input_size.py {model_name}".format(sys.argv[1]), shell=True, check=True, text=True, capture_output=True)
output = output.stdout

shapes = []
for line in output.split("\n"):
    line = line.rsplit(":", 1)
    if line[0] in inputs_list:
        line = line[1].split(",")
        shape_input = []
        for l in line:
            if l == ' ':
                shapes.append(shape_input)
                shape_input = []
            else:
                shape_input.append(int(l))
if shapes == []:
    shapes = [[1, 1, 1, 1]]

k = 0
for shape in shapes:
    calc_shape = 1
    shape2 = []
    for i in range(len(shape)):
        if shape[i] != 0:
            calc_shape *= shape[i]
            shape2.append(shape[i])
    numpy_array = numpy.ones(calc_shape, dtype=numpy.float32)
    numpy_array = numpy.array(numpy_array).reshape(shape2)

    tensor = numpy_helper.from_array(numpy_array)

    input_name = extract_filename(model_name) + "_" + str(k) + ".pb"
    print(input_name)
    k += 1
    with open(input_name, "wb") as f:
        f.write(tensor.SerializeToString()) 
