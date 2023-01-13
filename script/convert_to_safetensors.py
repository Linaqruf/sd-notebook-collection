import os
import argparse
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description='Convert a target model to output model')
parser.add_argument('-i', '--input-model', required=True, help='Path to the target model')
parser.add_argument('-p', '--pop-dict', action='store_true', help='Pop the state_dict from the model')
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'gpu'], help='Device to use for loading the model')

args = parser.parse_args()

input_model_path = args.input_model
pop_dict = args.pop_dict
device = args.device

if device == 'gpu':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

output_model_path = os.path.splitext(input_model_path)[0] + '.safetensors'

print("Loading model:", os.path.basename(input_model_path))
try:
    with torch.no_grad():
        print("Conversion in progress, please wait...")
        model_weights = torch.load(input_model_path, map_location=device)["state_dict"]
        if pop_dict:
            model_weights.pop("state_dict")
        save_file(model_weights, output_model_path)
    print(f'Successfully converted {os.path.basename(input_model_path)} to {os.path.basename(output_model_path)}')
    print(f'located in this path : {output_model_path}')
except Exception as ex:
    print(f'ERROR converting {os.path.basename(input_model_path)}: {ex}')

print('Done!')
