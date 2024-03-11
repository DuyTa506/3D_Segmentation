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
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_determinism(seed=0)

os.system('cls||clear')
print("Training ...")

data_dir = "/app/brats_2021_task1"
json_list = "/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/info.json"
roi = (128, 128, 128)
batch_size = 1
fold = 1
max_epochs = 100
val_every = 10
train_loader, val_loader,test_loader = get_loader(batch_size, data_dir, json_list, fold, roi, volume=1, test_size=0.2)
print("Done initialize dataloader !! ")

model = BRATS(use_VAE = True, train_loader = train_loader,val_loader = val_loader, test_loader=test_loader )
print("Loaded model ! ")

checkpoint_callback = ModelCheckpoint(
    monitor='val/MeanDiceScore',
    dirpath='./checkpoints/{}'.format("SegTransVAE"),
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
    'logs', 
    name = "SegTransVAE", 
    default_hp_metric = None 
)
print("Done load Callback")
trainer = pl.Trainer(#fast_dev_run = 10, 
#                     accelerator='ddp',
                    #overfit_batches=5,
                     devices = [0], 
                        precision=16,
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



