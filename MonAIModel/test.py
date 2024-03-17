import pytorch_lightning as pl
from trainer import BRATS
import os 
import torch
from dataset.utils import get_loader
from monai.inferers import sliding_window_inference
import csv
import math
from tqdm import tqdm
from monai import transforms
from metric import DiceScore, IoU ,hausdorff_metric, Sensitivity, Specificity,calculate_metrics,calculate_mean_metric



from monai.utils import set_determinism
set_determinism(seed=0)

os.system('cls||clear')
print("Testing ...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = '/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/3DUnet_Like/checkpoints/TransBTS/last.ckpt'
data_dir = "/app/brats_2021_task1"
json_list = "/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/info.json"
roi = (128, 128, 128)
batch_size = 1
fold = 1
infer_overlap = 0.5



post_transforms =   transforms.Compose(
                [transforms.EnsureType(),
                 transforms.Activations(sigmoid=True), 
                 transforms.AsDiscrete(threshold_values=True), 
                 ]
            )

def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )


    return _compute(input)
_, _,test_loader = get_loader(batch_size, data_dir, json_list, fold, roi, volume=1, test_size=0.2)
model = BRATS.load_from_checkpoint(CKPT, use_VAE = False, train_loader = None,val_loader = None, test_loader=test_loader ).eval()


csv_file = '/app/duy55/Unet-Like-3D-Medical-Image-Segmentation/metrics_test/TransBTS.csv'





mean_metrics = []

with torch.no_grad():
    for val_data in tqdm(test_loader):
        val_inputs = val_data["image"].to(device)
        labels = val_data['label'].to(device)
        test_outputs = inference(val_inputs)
        test_outputs = post_transforms(test_outputs)
        iou_tc, dice_tc, sensitivity_tc, specificity_tc, hausdorff_tc = calculate_metrics(y_pred=test_outputs[:, 0:1], y=labels[:, 0:1])
        iou_wt, dice_wt, sensitivity_wt, specificity_wt, hausdorff_wt = calculate_metrics(y_pred=test_outputs[:, 1:2], y=labels[:, 1:2])
        iou_et, dice_et, sensitivity_et, specificity_et, hausdorff_et = calculate_metrics(y_pred=test_outputs[:, 2:3], y=labels[:, 2:3])
        mean_iou = calculate_mean_metric(torch.stack([iou_tc, iou_wt, iou_et], dim=1))
        mean_dice = calculate_mean_metric(torch.stack([dice_tc, dice_wt, dice_et], dim=1))
        mean_sens = calculate_mean_metric(torch.stack([sensitivity_tc, sensitivity_wt, sensitivity_et], dim=1))
        mean_spec = calculate_mean_metric(torch.stack([specificity_tc, specificity_wt, specificity_et], dim=1))
        mean_hd95 = calculate_mean_metric(torch.stack([hausdorff_tc, hausdorff_wt, hausdorff_et], dim=1))
        if torch.isnan(mean_hd95):
            mean_hd95 = 0
        mean_metrics.append([
            mean_iou.item(), 
            mean_dice.item(), 
            mean_sens.item(), 
            mean_spec.item(), 
            float(mean_hd95)
        ])
        
mean_metrics_total = torch.tensor(mean_metrics).mean(dim=0)
mean_metrics_total = [round(metric, 4) for metric in mean_metrics_total.tolist()]


with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['Mean_IOU', 'Mean_Dice', 'Mean_Sensitivity', 'Mean_Specificity','Mean_HD95'])

    writer.writerow(mean_metrics_total)