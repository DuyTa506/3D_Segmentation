import os 
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import csv
import torch
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from models.SegTranVAE.SegTranVAE import SegTransVAE
from loss.loss import Loss_VAE, DiceScore
from monai.losses import DiceLoss
import pytorch_lightning as pl
from monai.inferers import sliding_window_inference





class BRATS(pl.LightningModule):
    def __init__(self,train_loader,val_loader,test_loader, use_VAE = True, lr = 1e-4 ):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.use_vae = use_VAE
        self.lr = lr
        self.model = SegTransVAE((128, 128, 128), 8, 4, 3, 768, 8, 4, 3072, in_channels_vae=128, use_VAE = use_VAE)

        self.loss_vae = Loss_VAE()
        self.dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
        self.post_trans_images = Compose(
                [EnsureType(),
                 Activations(sigmoid=True), 
                 AsDiscrete(threshold_values=True), 
                 ]
            )

        self.best_val_dice = 0
        
        self.training_step_outputs = []      
        self.val_step_loss = []              
        self.val_step_dice = []
        self.val_step_dice_tc = []              
        self.val_step_dice_wt = []
        self.val_step_dice_et = []              
        self.test_step_loss = []              
        self.test_step_dice = []
        self.test_step_dice_tc = []              
        self.test_step_dice_wt = []
        self.test_step_dice_et = [] 

    def forward(self, x, is_validation = True):
        return self.model(x, is_validation) 
    def training_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        
        if not self.use_vae:
            outputs = self.forward(inputs, is_validation=False)
            loss = self.dice_loss(outputs, labels)
        else:
            outputs, recon_batch, mu, sigma = self.forward(inputs, is_validation=False)
                   
            vae_loss = self.loss_vae(recon_batch, inputs, mu, sigma)
            dice_loss = self.dice_loss(outputs, labels)
            loss = dice_loss + 1/(4 * 128 * 128 * 128) * vae_loss
            self.training_step_outputs.append(loss)
            self.log('train/vae_loss', vae_loss)
            self.log('train/dice_loss', dice_loss)
            if batch_index == 10:

                tensorboard = self.logger.experiment  
                fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(10, 5))
                

                ax[0].imshow(inputs.detach().cpu()[0][0][:, :, 80], cmap='gray')
                ax[0].set_title("Input")

                ax[1].imshow(recon_batch.detach().cpu().float()[0][0][:,:, 80], cmap='gray')
                ax[1].set_title("Reconstruction")
                
                ax[2].imshow(labels.detach().cpu().float()[0][0][:,:, 80], cmap='gray')
                ax[2].set_title("Labels TC")
                
                ax[3].imshow(outputs.sigmoid().detach().cpu().float()[0][0][:,:, 80], cmap='gray')
                ax[3].set_title("TC")
                
                ax[4].imshow(labels.detach().cpu().float()[0][2][:,:, 80], cmap='gray')
                ax[4].set_title("Labels ET")
                
                ax[5].imshow(outputs.sigmoid().detach().cpu().float()[0][2][:,:, 80], cmap='gray')
                ax[5].set_title("ET")

                
                tensorboard.add_figure('train_visualize', fig, self.current_epoch)

        self.log('train/loss', loss)
        
        return loss
    
    def on_train_epoch_end(self):
        ## F1 Macro all epoch saving outputs and target per batch

        # free up the memory
        # --> HERE STEP 3 <--
        if len(self.training_step_outputs) > 0:
            epoch_average = torch.stack(self.training_step_outputs).mean()
            self.log("training_epoch_average", epoch_average)
            self.training_step_outputs.clear()  # free memory
        else:
            pass


    def validation_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
        roi_size = (128, 128, 128)
        sw_batch_size = 1
        outputs = sliding_window_inference(
                inputs, roi_size, sw_batch_size, self.model, overlap = 0.5)
        loss = self.dice_loss(outputs, labels)
        
      
        val_outputs = self.post_trans_images(outputs)
        
        
        metric_tc = DiceScore(y_pred=val_outputs[:, 0:1], y=labels[:, 0:1], include_background = True)
        metric_wt = DiceScore(y_pred=val_outputs[:, 1:2], y=labels[:, 1:2], include_background = True)
        metric_et = DiceScore(y_pred=val_outputs[:, 2:3], y=labels[:, 2:3], include_background = True)
        mean_val_dice =  (metric_tc + metric_wt + metric_et)/3
        self.val_step_loss.append(loss)           
        self.val_step_dice.append(mean_val_dice)
        self.val_step_dice_tc.append(metric_tc)              
        self.val_step_dice_wt.append(metric_wt)
        self.val_step_dice_et.append(metric_et) 
        return {'val_loss': loss, 'val_mean_dice': mean_val_dice, 'val_dice_tc': metric_tc,
                'val_dice_wt': metric_wt, 'val_dice_et': metric_et}
    
    def on_validation_epoch_end(self):

        loss = torch.stack(self.val_step_loss).mean()
        mean_val_dice = torch.stack(self.val_step_dice).mean()
        metric_tc = torch.stack(self.val_step_dice_tc).mean()
        metric_wt = torch.stack(self.val_step_dice_wt).mean()
        metric_et = torch.stack(self.val_step_dice_et).mean()
        self.log('val/Loss', loss)
        self.log('val/MeanDiceScore', mean_val_dice)
        self.log('val/DiceTC', metric_tc)
        self.log('val/DiceWT', metric_wt)
        self.log('val/DiceET', metric_et)
        os.makedirs(self.logger.log_dir,  exist_ok=True)
        if self.current_epoch == 0:
            with open('{}/metric_log.csv'.format(self.logger.log_dir), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Mean Dice Score', 'Dice TC', 'Dice WT', 'Dice ET'])
        with open('{}/metric_log.csv'.format(self.logger.log_dir), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.current_epoch, mean_val_dice.item(), metric_tc.item(), metric_wt.item(), metric_et.item()])

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
                f"\n Current epoch: {self.current_epoch} Current mean dice: {mean_val_dice:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\n Best mean dice: {self.best_val_dice}"
                f" at epoch: {self.best_val_epoch}"
            )
        
        self.val_step_loss.clear()           
        self.val_step_dice.clear()
        self.val_step_dice_tc.clear()             
        self.val_step_dice_wt.clear()
        self.val_step_dice_et.clear()
        return {'val_MeanDiceScore': mean_val_dice}
    def test_step(self, batch, batch_index):
        inputs, labels = (batch['image'], batch['label'])
    
        roi_size = (128, 128, 128)
        sw_batch_size = 1
        test_outputs = sliding_window_inference(
                    inputs, roi_size, sw_batch_size, self.forward, overlap = 0.5)
        loss = self.dice_loss(test_outputs, labels)
        test_outputs = self.post_trans_images(test_outputs)
        metric_tc = DiceScore(y_pred=test_outputs[:, 0:1], y=labels[:, 0:1], include_background = True)
        metric_wt = DiceScore(y_pred=test_outputs[:, 1:2], y=labels[:, 1:2], include_background = True)
        metric_et = DiceScore(y_pred=test_outputs[:, 2:3], y=labels[:, 2:3], include_background = True)
        mean_test_dice =  (metric_tc + metric_wt + metric_et)/3
        
        self.test_step_loss.append(loss)           
        self.test_step_dice.append(mean_test_dice)
        self.test_step_dice_tc.append(metric_tc)              
        self.test_step_dice_wt.append(metric_wt)
        self.test_step_dice_et.append(metric_et) 
    
        return {'test_loss': loss, 'test_mean_dice': mean_test_dice, 'test_dice_tc': metric_tc,
                'test_dice_wt': metric_wt, 'test_dice_et': metric_et}
    
    def test_epoch_end(self):
        loss = torch.stack(self.test_step_loss).mean()
        mean_test_dice = torch.stack(self.test_step_dice).mean()
        metric_tc = torch.stack(self.test_step_dice_tc).mean()
        metric_wt = torch.stack(self.test_step_dice_wt).mean()
        metric_et = torch.stack(self.test_step_dice_et).mean()
        self.log('test/Loss', loss)
        self.log('test/MeanDiceScore', mean_test_dice)
        self.log('test/DiceTC', metric_tc)
        self.log('test/DiceWT', metric_wt)
        self.log('test/DiceET', metric_et)

        with open('{}/test_log.csv'.format(self.logger.log_dir), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Mean Test Dice", "Dice TC", "Dice WT", "Dice ET"])
            writer.writerow([mean_test_dice, metric_tc, metric_wt, metric_et])

        self.test_step_loss.clear()           
        self.test_step_dice.clear()
        self.test_step_dice_tc.clear()             
        self.test_step_dice_wt.clear()
        self.test_step_dice_et.clear()
        return {'test_MeanDiceScore': mean_test_dice}
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                    self.model.parameters(), self.lr, weight_decay=1e-5, amsgrad=True
                    )
#         optimizer = AdaBelief(self.model.parameters(), 
#                             lr=self.lr, eps=1e-16, 
#                             betas=(0.9,0.999), weight_decouple = True, 
#                             rectify = False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return self.train_loader
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader