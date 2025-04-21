## simplify onnx

import onnx
from onnxsim import simplify
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    'input_onnx', type=str, help='input onnx')
parser.add_argument(
    'output_onnx', type=str, help='output onnx')
args = parser.parse_args()

onnx_model = onnx.load(args.input_onnx)
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, args.output_onnx)
