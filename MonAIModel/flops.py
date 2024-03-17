import argparse
from calflops import calculate_flops
import monai
parser = argparse.ArgumentParser(description='Calculate FLOPs for a given model.')
parser.add_argument('--model', type=str, default='UNet', help='Model name to calculate FLOPs.')
parser.add_argument('--input_shape', type=str, default='1,4,128,128,128', help='Input shape of the model.')
args = parser.parse_args()

input_shape = tuple(map(int, args.input_shape.split(',')))

if args.model == 'UNet': 
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
elif args.model == 'SwinUNet':
    model = monai.networks.nets.SwinUNETR(
    img_size=(128,128,128),
    in_channels=4,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
)
elif args.model == 'UNETR':
    model = monai.networks.nets.UNETR(
    in_channels=4,
    out_channels=3,
    img_size=(128,128,128),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
)
else:
    raise ValueError(f"Invalid model_type: {args.model}")

flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)


print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")