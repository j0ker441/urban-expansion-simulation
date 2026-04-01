import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import os
import tifffile


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1):
    input_tensor = image.cuda().float()

    net.eval()
    with torch.no_grad():
        outputs = net(input_tensor)

        n, c, h, w = outputs.size()
        temp_outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        probabilities = torch.softmax(temp_outputs, dim=-1).cpu().numpy()
        probabilities = probabilities.transpose(1, 0)  # (num_classes, h*w)
        probabilities = np.reshape(probabilities, (c, h, w))

        prediction = torch.argmax(outputs, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    label = label.squeeze().cpu().numpy().astype(np.uint8)

    metric_list = []
    for i in range(1, classes):
        pred_mask = (prediction == i)
        gt_mask = (label == i)
        dice, hd95 = calculate_metric_percase(pred_mask, gt_mask)
        metric_list.append((dice, hd95))

    try:
        os.makedirs(test_save_path, exist_ok=True)
        print(f"成功创建目录: {test_save_path}")
    except Exception as e:
        print(f"创建目录失败: {e}")
        return metric_list

    save_path = os.path.join(test_save_path, f"{case}.npy")
    try:
        np.save(save_path, probabilities)
        print(f"成功保存文件: {save_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")

    return metric_list

def calculate_metric_percase(pred, gt):
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)

    assert pred.ndim == 2 and gt.ndim == 2, f"需要二维输入，得到 pred:{pred.ndim}D, gt:{gt.ndim}D"

    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    if np.sum(gt) == 0:
        return 0, 0

    try:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return 0, 0

    return dice, hd95
