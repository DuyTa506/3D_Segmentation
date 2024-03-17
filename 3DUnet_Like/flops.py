import argparse
from calflops import calculate_flops
from models.SegTranVAE.SegTranVAE import SegTransVAE

parser = argparse.ArgumentParser(description='Calculate FLOPs for a given model.')
parser.add_argument('--model', type=str, default='SegTransVAE', help='Model name to calculate FLOPs.')
parser.add_argument('--input_shape', type=str, default='1,4,128,128,128', help='Input shape of the model.')
args = parser.parse_args()

input_shape = tuple(map(int, args.input_shape.split(',')))

if args.model == 'SegTransVAE':
    model = SegTransVAE((128, 128, 128), 8, 4, 3, 768, 8, 4, 3072, in_channels_vae=128, use_VAE=True)
elif args.model == 'TransBTS':
    model = SegTransVAE((128, 128, 128), 8, 4, 3, 768, 8, 4, 3072, in_channels_vae=128, use_VAE=False)
else :
    
    raise ValueError(f"Model {args.model} is not supported.")

flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)

print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")