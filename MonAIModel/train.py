import torch
import os
from monai.utils import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os 
from pytorch_lightning.loggers import TensorBoardLogger
from trainer import BRATS
from dataset.utils import get_loader
import pytorch_lightning as pl
import torch
import monai
torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_determinism(seed=0)
import argparse
parser = argparse.ArgumentParser(description='Training script for medical image segmentation.')
parser.add_argument('--model_type', type=str, default='Unet', choices=['UNETR', 'SwinUNet', 'SegResNet', 'UNet'],
                    help='Choose the type of model architecture: UNETR, SwinUNet, SegResNet, UNet')

parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Epochs for training')
parser.add_argument('--volume', type=float, default=1, help='Volume of dataset for training')
parser.add_argument('--test_size', type=float, default=0.2, help='Test size for evaluation and testing')


args = parser.parse_args()

os.system('cls||clear')
print("Training ...")

data_dir = "/app/brats_2021_task1"
json_list = "/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/info.json"
roi = (128, 128, 128)
fold = 1
max_epochs = args.epochs
val_every = 10
train_loader, val_loader, test_loader = get_loader(args.batch_size, data_dir, json_list, fold, roi, volume=args.volume, test_size=args.test_size)
print("Done initialize dataloader !! ")

model_type = args.model_type
if model_type == 'UNet': 
    model = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
elif model_type == 'SwinUNet':
    model = monai.networks.nets.SwinUNETR(
    img_size=roi,
    in_channels=4,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
)
elif model_type == 'SegResNet':
    model = monai.networks.nets.SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
)
elif model_type == 'UNETR':
    model = monai.networks.nets.UNETR(
    in_channels=4,
    out_channels=3,
    img_size=roi,
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
    raise ValueError(f"Invalid model_type: {model_type}")

model = BRATS(train_loader = train_loader,val_loader = val_loader, test_loader=test_loader, model = model )


print("Loaded model ! ")

checkpoint_callback = ModelCheckpoint(
    monitor='val/MeanDiceScore',
    dirpath='/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/checkpoints/{}'.format(args.model_type),
    filename='Epoch{epoch:3d}-MeanDiceScore{val/MeanDiceScore:.4f}',
    save_top_k=3,
    mode='max',
    save_last= True,
    auto_insert_metric_name=False
)
early_stop_callback = EarlyStopping(
   monitor='val/MeanDiceScore',
   min_delta=0.0001,
   patience=15,
   verbose=False,
   mode='max'
)
tensorboardlogger = TensorBoardLogger(
    '/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/logs', 
    name = "{}".format(args.model_type), 
    default_hp_metric = None 
)
print("Done load Callback")
trainer = pl.Trainer(#fast_dev_run = 10, 
#                     accelerator='ddp',
                    #overfit_batches=5,
                     devices = [0], 
                        precision="16-mixed",
                     max_epochs = max_epochs, 
                     enable_progress_bar=True,  
                     callbacks=[checkpoint_callback, early_stop_callback], 
#                     auto_lr_find=True,
                    num_sanity_val_steps=1,
                    logger = tensorboardlogger,
                    check_val_every_n_epoch = 10,
#                     limit_train_batches=0.01, 
#                     limit_val_batches=0.01
                     )

print("Done init trainer !")
# trainer.tune(model)
trainer.fit(model)



