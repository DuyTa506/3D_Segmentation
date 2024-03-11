import pytorch_lightning as pl
from trainer import BRATS
import os 
import torch
from dataset.utils import get_loader
os.system('cls||clear')
print("Testing ...")

CKPT = ''

train_loader, val_loader,test_loader = get_loader(batch_size, data_dir, json_list, fold, roi, volume=1, test_size=0.2)

model = BRATS(use_VAE = True, train_loader = train_loader,val_loader = val_loader, test_loader=test_loader ).load_from_checkpoint(CKPT).eval()

# val_dataloader = get_val_dataloader()
# test_dataloader = get_test_dataloader()
trainer = pl.Trainer(gpus = [0], precision=32, progress_bar_refresh_rate=10)

trainer.test(model, dataloaders = val_loader)
trainer.test(model, dataloaders = test_loader)