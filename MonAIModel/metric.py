from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric
from monai.metrics.utils import do_metric_reduction, ignore_background

def IoU(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
) -> torch.Tensor:
    """Computes Intersection over Union (IoU) metric from full size Tensor.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute IoU. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip IoU computation on the first channel of
            the predicted output. Defaults to True.
    Returns:
        IoU scores per batch and per class, (shape [batch_size, num_classes]).
    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    
    # Calculate intersection and union
    intersection = torch.sum(y * y_pred, dim=reduce_axis)
    union = torch.sum(y + y_pred, dim=reduce_axis) - intersection

    return torch.where(
        union > 0,
        intersection / union,
        torch.tensor(float("1"), device=intersection.device),
    )


def DiceScore(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = False,
) -> torch.Tensor:
    """Computes Dice score metric from full size Tensor and collects average.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
    Returns:
        Dice scores per batch and per class, (shape [batch_size, num_classes]).
    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)
 
    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o

    return torch.where(
        denominator > 0,
        (2.0 * intersection) / denominator,
        torch.tensor(float("1"), device=y_o.device),
    )

def Sensitivity(y_pred: torch.Tensor, y: torch.Tensor, include_background: bool = True) -> torch.Tensor:
    """Computes sensitivity from full size Tensor.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute sensitivity. It must be one-hot format and first dim is batch.
            The values should be binarized.
    Returns:
        Sensitivity scores per batch and per class, (shape [batch_size, num_classes]).
    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """
    # Calculate true positives and false negatives
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    
    intersection = torch.sum(y * y_pred, dim=reduce_axis)
    y_o = torch.sum(y, dim=reduce_axis)
    sensitivity = torch.where(
        y_o > 0,
        intersection / y_o,
        torch.tensor(float("1"), device=intersection.device),
    )
    return sensitivity

def Specificity(y_pred: torch.Tensor, y: torch.Tensor, include_background: bool = False) -> torch.Tensor:
    """Computes specificity from full size Tensor.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute specificity. It must be one-hot format and first dim is batch.
            The values should be binarized.
    Returns:
        Specificity scores per batch and per class, (shape [batch_size, num_classes]).
    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """
    # Calculate true negatives and false positives
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y = y.float()
    y_pred = y_pred.float()
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    true_negatives = torch.sum((1 - y_pred) * (1 - y), dim=reduce_axis)
    false_positives = torch.sum(y_pred * (1 - y), dim=reduce_axis)

    specificity = true_negatives / (true_negatives + false_positives)

    return specificity

hausdorff_metric = HausdorffDistanceMetric(include_background=True,distance_metric ="euclidean", reduction='mean', percentile=95,get_not_nans=False)


def calculate_metrics(y_pred, y):
    iou = IoU(y_pred, y)
    dice = DiceScore(y_pred, y)
    sensitivity = Sensitivity(y_pred, y)
    specificity = Specificity(y_pred, y)
    hausdorff = hausdorff_metric(y_pred, y)
    return iou, dice, sensitivity, specificity, hausdorff


def calculate_mean_metric(metrics):
    """Calculate mean metric while avoiding division by zero."""
    non_zero_count = torch.sum(metrics != 0, dim=1)
    mean_metric = torch.sum(metrics, dim=1) / non_zero_count
    mean_metric[non_zero_count == 0] = 0  # Set mean to 0 if all values are 0
    return mean_metric