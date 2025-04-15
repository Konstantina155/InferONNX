import onnx
import sys

if len(sys.argv) != 2:
    print("Usage: python3 onnx_get_io.py <onnx_model>")
    exit(1)

print (sys.argv[1])

model = onnx.load(sys.argv[1])

output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)